#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

extern void checkCUDA(cudaError_t error);

static cudaEvent_t start, stop;
static cudaEvent_t loc_start, loc_stop;

extern "C" void init_timer(void)
{
  checkCUDA(cudaEventCreate(&start));
  checkCUDA(cudaEventCreate(&stop));
}

extern "C" void init_local_timer(void)
{
  checkCUDA(cudaEventCreate(&loc_start));
  checkCUDA(cudaEventCreate(&loc_stop));
}

extern "C" void start_timer(void)
{
  checkCUDA(cudaEventRecord(start, 0));
}

extern "C" void stop_timer(float* ms)
{
  checkCUDA(cudaEventRecord(stop, 0));
  checkCUDA(cudaEventSynchronize(stop));
  checkCUDA(cudaEventElapsedTime(ms, start, stop));
}

extern "C" void start_local_timer(void)
{
  checkCUDA(cudaEventRecord(loc_start, 0));
}

extern "C" void stop_local_timer(float* ms)
{
  checkCUDA(cudaEventRecord(loc_stop, 0));
  checkCUDA(cudaEventSynchronize(loc_stop));
  checkCUDA(cudaEventElapsedTime(ms, loc_start, loc_stop));
}

extern "C" void free_timer(void)
{
  checkCUDA(cudaEventDestroy(start));
  checkCUDA(cudaEventDestroy(stop));
}

extern "C" void free_local_timer(void)
{
  checkCUDA(cudaEventDestroy(loc_start));
  checkCUDA(cudaEventDestroy(loc_stop));
}
