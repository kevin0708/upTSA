/* Communication with a DPU via the MRAM. */
/* Populate the MRAM with a collection of bytes and request the DPUs to */
/* compute the checksums. */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "params.h"
#include "timer.h"
#include <stdint.h>

#ifndef DPU_BINARY
#define DPU_BINARY "ts_dpu"
#endif

#define BUFFER_SIZE (1 << 16) // remove

#define MAX_DATA_VAL 127

DTYPE *tSeries;
DTYPE *query  ;
DTYPE *AMean  ;
DTYPE *ASigma ;
DTYPE minHost;
DTYPE minHostIdx;

static DTYPE *create_test_file(unsigned int ts_elements, unsigned int query_elements)
{
  srand(0);

  for (uint64_t i = 0; i < ts_elements; i++)
  {
    tSeries[i] = i % MAX_DATA_VAL;
  }

  for (uint64_t i = 0; i < query_elements; i++)
  {
    query[i] = tSeries[i];
  }

  return tSeries;
}

static void compute_ts_statistics(unsigned int timeSeriesLength, unsigned int ProfileLength, unsigned int queryLength)
{
  double *ACumSum = malloc(sizeof(double) * timeSeriesLength);
  ACumSum[0] = tSeries[0];
  for (uint64_t i = 1; i < timeSeriesLength; i++)
    ACumSum[i] = tSeries[i] + ACumSum[i - 1];
  double *ASqCumSum = malloc(sizeof(double) * timeSeriesLength);
  ASqCumSum[0] = tSeries[0] * tSeries[0];
  for (uint64_t i = 1; i < timeSeriesLength; i++)
    ASqCumSum[i] = tSeries[i] * tSeries[i] + ASqCumSum[i - 1];
  double *ASum = malloc(sizeof(double) * ProfileLength);
  ASum[0] = ACumSum[queryLength - 1];
  for (uint64_t i = 0; i < timeSeriesLength - queryLength; i++)
    ASum[i + 1] = ACumSum[queryLength + i] - ACumSum[i];
  double *ASumSq = malloc(sizeof(double) * ProfileLength);
  ASumSq[0] = ASqCumSum[queryLength - 1];
  for (uint64_t i = 0; i < timeSeriesLength - queryLength; i++)
    ASumSq[i + 1] = ASqCumSum[queryLength + i] - ASqCumSum[i];
  double *AMean_tmp = malloc(sizeof(double) * ProfileLength);
  for (uint64_t i = 0; i < ProfileLength; i++)
    AMean_tmp[i] = ASum[i] / queryLength;
  double *ASigmaSq = malloc(sizeof(double) * ProfileLength);
  for (uint64_t i = 0; i < ProfileLength; i++)
    ASigmaSq[i] = ASumSq[i] / queryLength - AMean[i] * AMean[i];
  for (uint64_t i = 0; i < ProfileLength; i++)
  {
    ASigma[i] = sqrt(ASigmaSq[i]);
    AMean[i] = (DTYPE)AMean_tmp[i];
  }

  free(ACumSum);
  free(ASqCumSum);
  free(ASum);
  free(ASumSq);
  free(ASigmaSq);
  free(AMean_tmp);
}

static void streamp_omp(DTYPE *tSeries, DTYPE *AMean, DTYPE *ASigma, int ProfileLength,
                        DTYPE *query, int queryLength, DTYPE queryMean, DTYPE queryStdDeviation)
{
  minHost = INT32_MAX;
  minHostIdx = 0;
  int numTread = 16;
  DTYPE tempMinHostIdx[numTread];
  DTYPE tempMinHost[numTread];
  for (int i = 0; i < numTread; i++)
  {
    tempMinHost[i] = INT32_MAX;
    tempMinHostIdx[i] = 0;
  }
  omp_set_num_threads(numTread);

#pragma omp parallel
  {
    int threadId = omp_get_thread_num();
#pragma omp for
    for (int subseq = 0; subseq < ProfileLength; subseq++)
    {
      DTYPE dotprod = 0;
      for (int j = 0; j < queryLength; j++)
      {
        dotprod += tSeries[j + subseq] * query[j];
      }

      DTYPE distance = 2 * (queryLength -
                      (dotprod - queryLength * AMean[subseq] * queryMean) 
                      / (ASigma[subseq] * queryStdDeviation));

      if (distance < tempMinHost[threadId])
      {
        tempMinHost[threadId] = distance;
        tempMinHostIdx[threadId] = subseq;
      }
    }

    // printf("I am %d thread, idx %d val %d \n", threadId, tempMinHostIdx[threadId], tempMinHost[threadId]);
  }

  for (int i = 0; i < numTread; i++)
  {
    if (tempMinHost[i] < minHost)
    {
      minHost = tempMinHost[i];
      minHostIdx = tempMinHostIdx[i];
    }
  }
  printf("min idx %d val %d \n", minHostIdx, minHost);
}

void query_params(unsigned int query_length, DTYPE *query_mean, DTYPE *query_std)
{
  double queryMean = 0;
  for (unsigned i = 0; i < query_length; i++)
    queryMean += query[i];
  queryMean /= (double)query_length;
  *query_mean = (DTYPE)queryMean;

  double queryStdDeviation;
  double queryVariance = 0;
  for (unsigned i = 0; i < query_length; i++)
  {
    queryVariance += (query[i] - queryMean) * (query[i] - queryMean);
  }
  queryVariance /= (double)query_length;
  queryStdDeviation = sqrt(queryVariance);
  *query_std = (DTYPE)queryStdDeviation;
}

int main()
{
  // Allocate memory for the arrays
    tSeries = (DTYPE *)malloc((1LL << 31) * sizeof(DTYPE));
    query = (DTYPE *)malloc((1 << 15) * sizeof(DTYPE));
    AMean = (DTYPE *)malloc((1LL << 31) * sizeof(DTYPE));
    ASigma = (DTYPE *)malloc((1LL << 31) * sizeof(DTYPE));
  // Timer declaration
  Timer timer;

  // set the dpus
  struct dpu_set_t set, dpu;
  uint32_t nr_of_dpus;

  DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &set));
  DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_of_dpus));
  printf("number of DPUs : %d\n", nr_of_dpus);
  printf("number of TLs  : %d\n", NR_TASKLETS);

  unsigned long int ts_size = 4096*2530*128;//4096*2048*64;
  const unsigned int query_length = 256;
  if (ts_size % (nr_of_dpus * NR_TASKLETS * query_length))
    ts_size = ts_size + (nr_of_dpus * NR_TASKLETS * query_length - ts_size % (nr_of_dpus * NR_TASKLETS * query_length));
  // ts_size += query_length;
  printf("ts_size is %d, query_len is %d \n", ts_size, query_length); // at least nr_of_dpus * NR_TASKLETS * query_length

  // Create an input file with arbitrary data
  create_test_file((ts_size + query_length), query_length);
  compute_ts_statistics(ts_size + query_length, ts_size, query_length);
  printf("ts %d query %d \n", tSeries[2], query[2]);

  DTYPE query_mean = 0;
  DTYPE query_std = 0;
  query_params(query_length, &query_mean, &query_std);
  printf("query mean %u std %u \n", query_mean, query_std);

  uint32_t slice_per_dpu = ts_size / nr_of_dpus;

  unsigned int kernel = 0;
  dpu_arguments_t input_arguments = {ts_size, query_length, slice_per_dpu, query_mean, query_std, kernel};
  uint32_t mem_offset;

  dpu_result_t result;
  result.minValue = INT32_MAX;
  result.minIndex = 0;

  start(&timer, 1, 0);
  int32_t i = 0;
  // Transfer arguments
  DPU_FOREACH(set, dpu)
  {
    DPU_ASSERT(dpu_copy_to(dpu, "DPU_INPUT_ARGUMENTS", 0, (const void *)&input_arguments, sizeof(input_arguments)));
    i++;
  }

  start(&timer, 8, 0);
  // Boardcast query
  DPU_ASSERT(dpu_broadcast_to(set, "query", 0, &query[0], sizeof(DTYPE) * query_length, DPU_XFER_DEFAULT));
  stop(&timer, 8);

  // Transfer TS
  i = 0;
  DPU_FOREACH(set, dpu, i)
  {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &tSeries[i * slice_per_dpu]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "TS", 0, (slice_per_dpu + query_length) * sizeof(DTYPE), DPU_XFER_DEFAULT));

  // Transfer Mean
  i = 0;
  DPU_FOREACH(set, dpu, i)
  {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &AMean[i * slice_per_dpu]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "Mean", 0, slice_per_dpu * sizeof(DTYPE), DPU_XFER_DEFAULT));

  // Transfer sigma
  i = 0;
  DPU_FOREACH(set, dpu, i)
  {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &ASigma[i * slice_per_dpu]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_TO_DPU, "Sigma", 0, slice_per_dpu * sizeof(DTYPE), DPU_XFER_DEFAULT));
  printf("finish transfer\n");
  stop(&timer, 1);

  // Run kernel on DPUs
  start(&timer, 2, 0);
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
  stop(&timer, 2);

  // dpu_result_t *results_retrieve[nr_of_dpus];
  DTYPE minVal[nr_of_dpus];
  DTYPE minIdx[nr_of_dpus];

  // retrieve partial results
  start(&timer, 3, 0);
  i = 0;
  DPU_FOREACH(set, dpu, i)
  {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &minVal[i]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "finalMinVal", 0, sizeof(DTYPE), DPU_XFER_DEFAULT));

  i = 0;
  DPU_FOREACH(set, dpu, i)
  {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &minIdx[i]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "finalMinIdx", 0, sizeof(DTYPE), DPU_XFER_DEFAULT));

  i = 0;
  DPU_FOREACH(set, dpu, i)
  {
    // printf("%d %d \n", minVal[i], minIdx[i]);
    if (result.minValue - minVal[i] > 0 && minVal[i] != 0)
    {
      result.minValue = minVal[i];
      result.minIndex = minIdx[i] + (i * slice_per_dpu);
    }
    // printf("res  val %d idx %d \n", result.minValue, result.minIndex);
    i++;
  }
  result.minValue *= 2;
  stop(&timer, 3);
  
  omp_set_num_threads(16);
  start(&timer, 4, 0);
  streamp_omp(tSeries, AMean, ASigma, ts_size - query_length - 1, query, query_length, query_mean, query_std);
  stop(&timer, 4);

  // printf("LOGS\n");
  // DPU_FOREACH(set, dpu)
  // {
  //   DPU_ASSERT(dpu_log_read(dpu, stdout));
  // }

  // Print timing results
  printf("CPU Version Time (ms): ");
  print(&timer, 4, 1);
  printf("Total CPU-DPU Time (ms): ");
  print(&timer, 1, 1);
  printf("Query boradcast Time (ms): ");
  print(&timer, 8, 1);
  printf("DPU Kernel Time (ms): ");
  print(&timer, 2, 1);
  printf("DPU-CPU Time (ms): ");
  print(&timer, 3, 1);

  // printf("\ncpu result is idx:%d val:%d \n", minHostIdx, minHost);
  // printf("dpu result is idx:%d val:%d \n", result.minIndex, result.minValue);
  int status = ((minHost == result.minValue) && (minHostIdx == result.minIndex));
  if (status)
  {
    printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] results are equal\n");
  }
  else
  {
    printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] results differ!\n");
  }
//---------------------------------------------------------------------------------------------------------------------------------
  
  for (uint64_t i = 0; i < query_length; i++)
  {
    query[i] = tSeries[i + 1];
  }
  // Boardcast query
  DPU_ASSERT(dpu_broadcast_to(set, "query", 0, &query[0], sizeof(DTYPE) * query_length, DPU_XFER_DEFAULT));

  kernel = 1;
  dpu_arguments_t input_arguments2 = {ts_size, query_length, slice_per_dpu, query_mean, query_std, kernel};

  i = 0;
  // Transfer arguments
  DPU_FOREACH(set, dpu)
  {
    DPU_ASSERT(dpu_copy_to(dpu, "DPU_INPUT_ARGUMENTS", 0, (const void *)&input_arguments2, sizeof(input_arguments)));
    i++;
  }


  start(&timer, 5, 0);
  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));
  stop(&timer, 5);

  start(&timer, 6, 0);
  i = 0;
  DPU_FOREACH(set, dpu, i)
  {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &minVal[i]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "finalMinVal", 0, sizeof(DTYPE), DPU_XFER_DEFAULT));

  i = 0;
  DPU_FOREACH(set, dpu, i)
  {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &minIdx[i]));
  }
  DPU_ASSERT(dpu_push_xfer(set, DPU_XFER_FROM_DPU, "finalMinIdx", 0, sizeof(DTYPE), DPU_XFER_DEFAULT));

  result.minValue = INT32_MAX;
  result.minIndex = 0;
  i = 0;
  DPU_FOREACH(set, dpu, i)
  {
    // printf("%d %d \n", minVal[i], minIdx[i]);
    if (result.minValue - minVal[i] > 0 && minVal[i] != 0)
    {
      result.minValue = minVal[i];
      result.minIndex = minIdx[i] + (i * slice_per_dpu);
    }
    // printf("res  val %d idx %d \n", result.minValue, result.minIndex);
    i++;
  }
  result.minValue *= 2;
  stop(&timer, 6);

  start(&timer, 7, 0);
  streamp_omp(tSeries, AMean, ASigma, ts_size - query_length - 1, query, query_length, query_mean, query_std);
  stop(&timer, 7);

  // printf("\ncpu result is idx:%d val:%d \n", minHostIdx, minHost);
  // printf("dpu result is idx:%d val:%d \n", result.minIndex, result.minValue);

  // Print timing results
  printf("CPU Version Time (ms): ");
  print(&timer, 7, 1);
  printf("DPU Kernel Time (ms): ");
  print(&timer, 5, 1);
  printf("DPU-CPU Time (ms): ");
  print(&timer, 6, 1);

  status = ((minHost == result.minValue) && (minHostIdx == result.minIndex));
  if (status)
  {
    printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] results are equal\n");
  }
  else
  {
    printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] results differ!\n");
  }

  // printf("LOGS\n");
  // DPU_FOREACH(set, dpu)
  // {
  //   DPU_ASSERT(dpu_log_read(dpu, stdout));
  // }
  // printf("dpu result is idx:%d val:%d \n", result.minIndex, result.minValue);

  DPU_ASSERT(dpu_free(set));
  
  // Free the allocated memory
    free(tSeries);
    free(query);
    free(AMean);
    free(ASigma);
  return 0;
}