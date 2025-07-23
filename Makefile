all:
	dpu-upmem-dpurte-clang -DNR_TASKLETS=16 dpu.c -o ts_dpu -Isupport -O3
	gcc -O3 --std=c99 -o ts_host host.c -g `dpu-pkg-config --cflags --libs dpu` -lm -fopenmp -Isupport -DNR_DPUS=2530 -DNR_TASKLETS=16