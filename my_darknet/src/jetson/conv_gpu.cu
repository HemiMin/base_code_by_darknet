#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "cublas_v2.h"

#include "type.h"

void checkCUDA(cudaError_t error)
{
  if (error != (cudaError_t)CUDA_SUCCESS) {
    std::cerr << "[ERROR] CUDA " << error << std::endl;
    exit(0);
  }
}

void checkCuBLAS(cublasStatus_t status)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "[ERROR] cuBLAS ";
    switch (status) {
      case CUBLAS_STATUS_NOT_INITIALIZED:  std::cerr << "not initialized";  break;
      case CUBLAS_STATUS_ALLOC_FAILED:     std::cerr << "alloc failed";     break;
      case CUBLAS_STATUS_INVALID_VALUE:    std::cerr << "invalid value";    break;
      case CUBLAS_STATUS_ARCH_MISMATCH:    std::cerr << "arch mismatch";    break;
      case CUBLAS_STATUS_MAPPING_ERROR:    std::cerr << "mapping error";    break;
      case CUBLAS_STATUS_EXECUTION_FAILED: std::cerr << "execution failed"; break;
      case CUBLAS_STATUS_INTERNAL_ERROR:   std::cerr << "internal error";   break;
      case CUBLAS_STATUS_NOT_SUPPORTED:    std::cerr << "not supported";    break;
      case CUBLAS_STATUS_LICENSE_ERROR:    std::cerr << "license error";    break;
      default:                             std::cerr << "unknown error";    break;
    }
    std::cerr << std::endl;
    exit(0);
  }
}

#ifdef CUBLAS
extern "C" void cublas_sgemm(int M, int N, int K, scalar_t* A, int lda, scalar_t* B, int ldb, scalar_t* C, int ldc)
{
  cublasHandle_t handle;
  scalar_t *A_d, *B_d, *C_d;

  checkCuBLAS(cublasCreate(&handle));

  checkCUDA(cudaMalloc((void**)&A_d, sizeof(scalar_t)*N*K));
  checkCUDA(cudaMalloc((void**)&B_d, sizeof(scalar_t)*K*M));
  checkCUDA(cudaMalloc((void**)&C_d, sizeof(scalar_t)*N*M));

  const float alpha = 1.0f;
  const float beta = 1.0f;

  checkCuBLAS(cublasSetMatrix(N, K, sizeof(scalar_t), B, ldb, A_d, ldb));
  checkCuBLAS(cublasSetMatrix(K, M, sizeof(scalar_t), A, lda, B_d, lda));
  checkCuBLAS(cublasSetMatrix(N, M, sizeof(scalar_t), C, ldc, C_d, ldc));

  checkCuBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, A_d, ldb, B_d, lda, &beta, C_d, ldc));

  checkCuBLAS(cublasGetMatrix(N, M, sizeof(scalar_t), C_d, ldc, C, ldc));

  checkCUDA(cudaFree(A_d));
  checkCUDA(cudaFree(B_d));
  checkCUDA(cudaFree(C_d));

  cublasDestroy(handle);
}

void cublas_sgemm_except_memcpy(int M, int N, int K, scalar_t* A, int lda, scalar_t* B, int ldb, scalar_t* C, int ldc)
{
  cublasHandle_t handle;
  checkCuBLAS(cublasCreate(&handle));
  const float alpha = 1.0f;
  const float beta = 1.0f;

  checkCuBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc));

  cublasDestroy(handle);
}
#endif

__global__ void conv2d_cuda(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
                            int out_ch, int in_ch, int k_size, int in_h, int in_w,
                            int stride, int l_pad, int r_pad, int u_pad, int d_pad)
{
  int out_w = (in_w + l_pad+r_pad - k_size) / stride + 1;
  int out_h = (in_h + u_pad+d_pad - k_size) / stride + 1;

  if (blockIdx.x < out_w*out_h) { // N
    if (blockIdx.y < out_ch) { // M
      if (threadIdx.x < in_ch*k_size*k_size) { // K
        int w_offset = threadIdx.x % k_size;
        int h_offset = (threadIdx.x / k_size) % k_size;
        int c_offset = threadIdx.x / k_size / k_size;
        int ot_w_idx = blockIdx.x % out_w;
        int ot_h_idx = blockIdx.x / out_w;
        int w = ot_w_idx*stride + w_offset - l_pad;
        int h = ot_h_idx*stride + h_offset - u_pad;
        int in_idx = w + in_w*(h + in_h*c_offset);
        int wt_idx = threadIdx.x + in_ch*k_size*k_size*blockIdx.y;
        int ot_idx = blockIdx.x + out_w*out_h*blockIdx.y;
        if (w >= 0 && w < in_w && h >= 0 && h < in_h) {
          OUTPUT[ot_idx] += INPUT[in_idx] * WEIGHT[wt_idx];
        }
      }
    }
  }
  /*
  if (threadIdx.x < out_ch) { // M
    if (blockIdx.y < in_ch*k_size*k_size) { // K
      if (blockIdx.x < out_w*out_h) { // N
        int w_offset = blockIdx.y % k_size;
        int h_offset = (blockIdx.y / k_size) % k_size;
        int c_offset = blockIdx.y / k_size / k_size;
        int ot_w_idx = blockIdx.x % out_w;  // output w index
        int ot_h_idx = blockIdx.x / out_w;  // output h index
        int w = ot_w_idx*stride + w_offset - l_pad;  // input w index
        int h = ot_h_idx*stride + h_offset - u_pad;  // input h index
        int in_idx = w + in_w*(h + in_h*c_offset);
        int wt_idx = blockIdx.y + in_ch*k_size*k_size*threadIdx.x;
        int ot_idx = blockIdx.x + out_w*out_h*threadIdx.x;
        if (w >= 0 && w < in_w && h >= 0 && h < in_h) {
          OUTPUT[ot_idx] += INPUT[in_idx] * WEIGHT[wt_idx];
        }
      }
    }
  }
*/
}

__global__ void im2col_gpu(scalar_t* data_im, scalar_t* data_col, int channel, int height, int width, int ksize,
                           int stride, int l_pad, int r_pad, int u_pad, int d_pad)
{
  /*
  int height_col = (height + u_pad+d_pad - ksize) / stride + 1;
  int width_col  = (width  + l_pad+r_pad - ksize) / stride + 1;

  if (blockIdx.y < channel * ksize * ksize) {
    if (blockIdx.x < width_col * height_col) {
      int w_offset = blockIdx.y % ksize;
      int h_offset = (blockIdx.y / ksize) % ksize;
      int c_im = blockIdx.y / ksize / ksize;
      int w = blockIdx.x % width_col;
      int h = blockIdx.x / width_col;
      int im_row = h_offset + h*stride - u_pad;
      int im_col = w_offset + w*stride - l_pad;
      if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
        data_col[blockIdx.x + width_col*height_col*blockIdx.y] = data_im[im_col + width*(im_row + height*c_im)];
      } else {
        data_col[blockIdx.x + width_col*height_col*blockIdx.y] = 0;
      }

    }
  }
  */
  int k_w, k_h;
  for (k_h = 0; k_h < ksize; ++k_h){
    for (k_w = 0; k_w < ksize; ++k_w){
      int col_index = gridDim.x*gridDim.y*ksize*ksize*threadIdx.x + gridDim.x*gridDim.y*(ksize*k_h+k_w)
        + gridDim.x*blockIdx.y + blockIdx.x;
      int im_row = k_h + blockIdx.y*stride;
      int im_col = k_w + blockIdx.x*stride;
      int row = im_row - u_pad;
      int col = im_col - l_pad;
      if (row < 0 || col < 0 || row >= height || col >= width) {
        data_col[col_index]=0;
      }
      else {
        data_col[col_index] = data_im[col+width*(row+height*threadIdx.x)];
      }
    }
  }
}

extern "C" void conv2d_gpu(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
                           int out_ch, int in_ch, int k_size, int in_h, int in_w,
                           int stride, int l_pad, int r_pad, int u_pad, int d_pad)
{
  int out_w = (in_w + l_pad+r_pad - k_size) / stride + 1;
  int out_h = (in_h + u_pad+d_pad - k_size) / stride + 1;

#ifdef CUDA
  conv2d_cuda<<<dim3(out_w*out_h, in_ch*k_size*k_size), out_ch>>>(INPUT, WEIGHT, OUTPUT,
                                                                  out_ch, in_ch, k_size, in_h, in_w,
                                                                  stride, l_pad, r_pad, u_pad, d_pad);
#elif defined(CUBLAS)
  scalar_t* data_col;
  checkCUDA(cudaMalloc((void**)&data_col, sizeof(scalar_t)*out_w*out_h*in_ch*k_size*k_size));
  //im2col_gpu<<<dim3(out_w*out_h, in_ch*k_size*k_size), 1>>>(INPUT, data_col, in_ch, in_h, in_w, k_size, stride, l_pad, r_pad, u_pad, d_pad);
  im2col_gpu<<<dim3(out_w, out_h), in_ch>>>(INPUT, data_col, in_ch, in_h, in_w, k_size, stride, l_pad, r_pad, u_pad, d_pad);

  int M = out_ch, N = out_h*out_w, K = in_ch*k_size*k_size;
  cublas_sgemm_except_memcpy(M, N, K, WEIGHT, K, data_col, N, OUTPUT, N);

  checkCUDA(cudaFree(data_col));
#endif

}

extern "C" void conv2d_gpu_except_memcpy( scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
                                          int out_ch, int in_ch, int k_size, int in_h, int in_w,
                                          int stride, int l_pad, int r_pad, int u_pad, int d_pad)
{
  int out_w = (in_w + l_pad+r_pad - k_size) / stride + 1;
  int out_h = (in_h + u_pad+d_pad - k_size) / stride + 1;

  conv2d_cuda<<<dim3(out_w*out_h, in_ch*k_size*k_size), out_ch>>>(INPUT, WEIGHT, OUTPUT,
                                                                  out_ch, in_ch, k_size, in_h, in_w,
                                                                  stride, l_pad, r_pad, u_pad, d_pad);
}
