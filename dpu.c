#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <mram.h>
#include <barrier.h>
#include <perfcounter.h>
#include <string.h>
#include "common.h"

#define ELEM_IN_BLOCK BLOCK_SIZE / sizeof(DTYPE)
#define CACHE_SIZE ELEM_IN_BLOCK + 2
#define BUFFER_SIZE (1 << 20) // 2M

__mram DTYPE TS[BUFFER_SIZE];
__mram DTYPE Mean[BUFFER_SIZE];  // all ts mean
__mram DTYPE Sigma[BUFFER_SIZE]; // all sigma
__mram DTYPE Qs[BUFFER_SIZE];

__host __dma_aligned DTYPE query[1024];

__host __dma_aligned DTYPE query_last_TS;
// result 不变
__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host DTYPE minVal[NR_TASKLETS];
__host DTYPE minIdx[NR_TASKLETS];

__host DTYPE finalMinVal;
__host DTYPE finalMinIdx;

extern int main_kernel1(void);
extern int main_kernel2(void);

int (*kernels[2])(void) = {main_kernel1, main_kernel2};

int main(void)
{
  // Kernel
  return kernels[DPU_INPUT_ARGUMENTS.kernel]();
}

BARRIER_INIT(my_barrier, NR_TASKLETS);
int main_kernel1()
{

  int tasklet_id = me();
  if (tasklet_id == 0) // id = 0的tasklet要重置一下mem
  {
    mem_reset(); // Reset the heap
  }
  // Input arguments,定义变量然后从inputargs里面取对应的赋值.
  DTYPE ts_size = DPU_INPUT_ARGUMENTS.ts_length;
  DTYPE query_length = DPU_INPUT_ARGUMENTS.query_length;
  DTYPE profilelength = ts_size - query_length;

  DTYPE slice_per_dpu = DPU_INPUT_ARGUMENTS.slice_per_dpu;
  DTYPE query_mean = DPU_INPUT_ARGUMENTS.query_mean;
  DTYPE query_std = DPU_INPUT_ARGUMENTS.query_std;
  // printf("BL %d\n", BLOCK_SIZE);
  // printf("%d %d %d %d %d %d\n", ts_size, query_length, profilelength, slice_per_dpu, query_mean, query_std);

  // DTYPE myoffset = dpu_id * slice_per_dpu;
  DTYPE my_slice = slice_per_dpu / NR_TASKLETS;
  if (my_slice % 2)
    my_slice = my_slice + 1;
  DTYPE my_start = tasklet_id * my_slice;
  DTYPE my_end = my_start + my_slice - 1;
  // printf("mystart %d myend %d\n", my_start, my_end);
  // create caches
  DTYPE *cache_TS = (DTYPE *)mem_alloc(6 * BLOCK_SIZE);
  DTYPE *cache_mean = (DTYPE *)mem_alloc(BLOCK_SIZE);
  DTYPE *cache_sigma = (DTYPE *)mem_alloc(BLOCK_SIZE);
  DTYPE *cache_Qs = (DTYPE *)mem_alloc(BLOCK_SIZE);
  // DTYPE cache_TS   [2*ELEM_IN_BLOCK];
  // DTYPE cache_mean [ELEM_IN_BLOCK];
  // DTYPE cache_sigma[ELEM_IN_BLOCK];
  // DTYPE cache_Qs   [ELEM_IN_BLOCK];

  DTYPE min_distance = 10000;
  uint32_t min_index = 0;
  minVal[tasklet_id] = INT32_MAX;

  barrier_wait(&my_barrier);
  // process the data
  for (int i = my_start; i < my_end; i += ELEM_IN_BLOCK)
  {
    for (int x = 0; x < ELEM_IN_BLOCK; x++)
    {
      cache_Qs[x] = 0;
    }
    // barrier_wait(&my_barrier);
    mram_read(&TS[i], cache_TS, (ELEM_IN_BLOCK + query_length) * sizeof(DTYPE));
    for (int32_t j = 0; j < ELEM_IN_BLOCK; j++)
    {
      // calculate dotproduct
      #pragma unroll 8
      for (int32_t x = 0; x < query_length; x++)
      {
        cache_Qs[j] += cache_TS[x + j] * query[x];
        // if(j==10 && tasklet_id == 0)printf("id0 cacheTS %d query%d cache_Q %d\n", cache_TS[x+j],query[x],cache_Qs[j]);
      }
    }

    mram_read(&Mean[i], cache_mean, BLOCK_SIZE);
    mram_read(&Sigma[i], cache_sigma, BLOCK_SIZE);
    for (int32_t j = 0; j < ELEM_IN_BLOCK; j++)
    {
      // printf("Q %u \n",cache_Qs[j]);
      // DTYPE sigma_prod = (cache_sigma[j] * query_std);
      // DTYPE QminMean = (cache_Qs[j] - query_length * cache_mean[j] * query_mean);
      // printf("QminMean %u sigma_prod %u\n", QminMean, sigma_prod);
      // DTYPE temp = QminMean / sigma_prod;
      // printf("temp %d\n", temp);
      DTYPE distance = query_length - (cache_Qs[j] - query_length * cache_mean[j] * query_mean) / (cache_sigma[j] * query_std);
      // if(tasklet_id == 0)printf("id %d distance  cacheQ %d cache_mean %d query_mean %d cache_sigma %d query_std %d\n",tasklet_id ,cache_Qs[j], cache_mean[j], query_mean, cache_sigma[j], query_std);

      if (distance < min_distance)
      {
        min_distance = distance;
        min_index = i + j;
      }
    }

    // save the Qs to mram
    mram_write(cache_Qs, &Qs[i], sizeof(DTYPE) * ELEM_IN_BLOCK);
  }
  barrier_wait(&my_barrier);
  minVal[tasklet_id] = min_distance;
  minIdx[tasklet_id] = min_index;
  // printf("val %u index %u\n", min_distance, min_index);
  if (tasklet_id == 0)
  {
    finalMinVal = 10000;
    finalMinIdx = 0;
    for (size_t i = 0; i < 16; i++)
    {
      if (minVal[i] < finalMinVal)
      {
        finalMinVal = minVal[i];
        finalMinIdx = minIdx[i];
      }
    }
    // printf("finalMinval indx %d val %d\n", finalMinIdx, finalMinVal);
  }
  if (tasklet_id == 0)
  {
    query_last_TS = query[0];
  }

  mram_read(&Qs[my_start], cache_Qs, BLOCK_SIZE);
  // printf("Qs %d \n",Qs[0]);
  barrier_wait(&my_barrier);
  return 0;
}

int main_kernel2()
{

  int tasklet_id = me();
  if (tasklet_id == 0) // id = 0的tasklet要重置一下mem
  {
    mem_reset(); // Reset the heap
  }

  // Input arguments,定义变量然后从inputargs里面取对应的赋值.
  DTYPE ts_size = DPU_INPUT_ARGUMENTS.ts_length;
  DTYPE query_length = DPU_INPUT_ARGUMENTS.query_length;
  DTYPE profilelength = ts_size - query_length;

  DTYPE slice_per_dpu = DPU_INPUT_ARGUMENTS.slice_per_dpu;
  DTYPE query_mean = DPU_INPUT_ARGUMENTS.query_mean;
  DTYPE query_std = DPU_INPUT_ARGUMENTS.query_std;
  // printf("BL %d\n", BLOCK_SIZE);
  // printf("%d %d %d %d %d %d\n", ts_size, query_length, profilelength, slice_per_dpu, query_mean, query_std);

  // DTYPE myoffset = dpu_id * slice_per_dpu;
  DTYPE my_slice = slice_per_dpu / NR_TASKLETS;
  if (my_slice % 2)
    my_slice = my_slice + 1;
  DTYPE my_start = tasklet_id * my_slice;
  DTYPE my_end = my_start + my_slice - 1;
  printf("mystart %d myend %d\n", my_start, my_end);

  // create caches
  DTYPE *cache_TS =   (DTYPE *)mem_alloc(4 * BLOCK_SIZE);
  DTYPE *cache_TS_next = (DTYPE *)mem_alloc(BLOCK_SIZE + 8);
  DTYPE *cache_mean = (DTYPE *)mem_alloc(BLOCK_SIZE + 8);
  DTYPE *cache_sigma = (DTYPE *)mem_alloc(BLOCK_SIZE + 8);
  DTYPE *cache_Qs =    (DTYPE *)mem_alloc(BLOCK_SIZE + 8);
  DTYPE *cache_rd_Qs = (DTYPE *)mem_alloc(BLOCK_SIZE);
  DTYPE *cache_wr_Qs = (DTYPE *)mem_alloc(BLOCK_SIZE);

  DTYPE cache_lastQ = 0;
  DTYPE cache_last_TS = 0;
  DTYPE cache_last_TS_next = 0;

  DTYPE min_distance = 10000;
  uint32_t min_index = 0;
  minVal[tasklet_id] = INT32_MAX;

  memset(cache_wr_Qs, 0, BLOCK_SIZE);
  memset(cache_rd_Qs, 0, BLOCK_SIZE);
  barrier_wait(&my_barrier); // this is a very important sync, do not modify it

  mram_read(&TS[my_start], cache_TS, query_length * sizeof(DTYPE));
  for (uint32_t k = 0; k < query_length; k++)
  {
    cache_lastQ += cache_TS[k] * query[k];
  }

  // process the data
  for (int i = my_start; i < my_end; i += ELEM_IN_BLOCK)
  {
    barrier_wait(&my_barrier);
    mram_read(&Qs[i], cache_rd_Qs, BLOCK_SIZE);
    // barrier_wait(&my_barrier);
    mram_read(&TS[i], cache_TS, ELEM_IN_BLOCK * sizeof(DTYPE));
    mram_read(&TS[i + query_length], cache_TS_next, BLOCK_SIZE);
    // printf("Qs %d \n",Qs[0]);
    // batch process Q
    cache_wr_Qs[0] = cache_lastQ + query[query_length - 1] * cache_last_TS_next - query_last_TS * cache_last_TS;
    // 	if(tasklet_id == 0)printf(" %d = %d + %d * %d - %d * %d \n",
    // cache_wr_Qs[0], cache_lastQ,  query[query_len_min_1], cache_last_TS_next, query_last_TS , cache_last_TS);
    // #pragma unroll 8
    for (uint32_t j = 0; j < ELEM_IN_BLOCK - 1; j++)
    {
      cache_wr_Qs[j + 1] = cache_rd_Qs[j] + query[query_length - 1] * cache_TS_next[j] - query_last_TS * cache_TS[j];
      // if(tasklet_id == 0)printf(" j= %d Q %d = %d + %d * %d - %d * %d \n",
      //     j+1, cache_wr_Qs[j+1], cache_rd_Qs[j],  query[query_length - 1], cache_TS_next[j], query_last_TS , cache_TS[j]);
    }
    cache_lastQ = cache_rd_Qs[ELEM_IN_BLOCK-1];
    cache_last_TS = cache_TS[ELEM_IN_BLOCK-1];
    cache_last_TS_next = cache_TS_next[ELEM_IN_BLOCK-1];

    mram_read(&Mean[i], cache_mean, BLOCK_SIZE);
    mram_read(&Sigma[i], cache_sigma, BLOCK_SIZE);
    for (int32_t j = 0; j < ELEM_IN_BLOCK; j++)
    {
      DTYPE distance = query_length - (cache_wr_Qs[j] - query_length * cache_mean[j] * query_mean) / (cache_sigma[j] * query_std);
      // if(tasklet_id == 0)printf("id %d %d distance %d cacheQ %d cache_mean %d query_mean %d cache_sigma %d query_std %d\n",tasklet_id, i+j,distance,cache_wr_Qs[j], cache_mean[j], query_mean, cache_sigma[j], query_std);

      if (distance < min_distance && distance >= 0)
      {
        min_distance = distance;
        min_index = i + j;
      }
    }

    // save the Qs to mram
    mram_write(cache_wr_Qs, &Qs[i], sizeof(DTYPE) * ELEM_IN_BLOCK);
  }

  barrier_wait(&my_barrier);
  minVal[tasklet_id] = min_distance;
  minIdx[tasklet_id] = min_index;
  // printf("val %u index %u\n", min_distance, min_index);
  if (tasklet_id == 0)
  {
    finalMinVal = 10000;
    finalMinIdx = 0;
    for (size_t i = 0; i < 16; i++)
    {
      if (minVal[i] < finalMinVal)
      {
        finalMinVal = minVal[i];
        finalMinIdx = minIdx[i];
      }
    }
    printf("finalMinval indx %d val %d\n", finalMinIdx, finalMinVal);
  }
  barrier_wait(&my_barrier);
  return 0;
}