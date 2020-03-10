#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "type.h"

extern void checkCUDA(cudaError_t error);

extern "C" void cudaMallocWrapper(void** ptr, size_t size)
{
  checkCUDA(cudaMalloc(ptr, size));
}

extern "C" void cudaFreeWrapper(scalar_t* ptr)
{
  checkCUDA(cudaFree(ptr));
}
