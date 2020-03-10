#include <stdlib.h>
#include <stdio.h>
#ifdef OPENBLAS
#include <cblas.h>
#endif

#include "ops.h"
#ifdef CUBLAS
#ifdef __cplusplus
extern "C" {
#endif
extern void cublas_sgemm(int M, int N, int K, scalar_t* A, int lda, scalar_t* B, int ldb, scalar_t* C, int ldc);
#ifdef __cplusplus
}
#endif
#endif

inline scalar_t im2col_get_pixel(scalar_t* im, int height, int width,
                                 int row, int col, int channel, 
                                 int l_pad, int u_pad)
{
  row -= u_pad;
  col -= l_pad;

  if (row < 0 || col < 0 ||
      row >= height || col >= width) return 0;
  return im[col + width*(row + height*channel)];
}

void im2col(scalar_t* data_im, scalar_t* data_col, int channel, int height, int width, int ksize,
            int stride, int l_pad, int r_pad, int u_pad, int d_pad)
{
  int c,h,w;
  int height_col = (height + u_pad+d_pad - ksize) / stride + 1;
  int width_col  = (width  + l_pad+r_pad - ksize) / stride + 1;

  int channel_col = channel * ksize * ksize;
  for (c = 0 ; c < channel_col ; ++c) {
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    for (h = 0 ; h < height_col ; ++h) {
      for (w = 0 ; w < width_col ; ++w) {
        int im_row = h_offset + h*stride;
        int im_col = w_offset + w*stride;
        int col_index = (c*height_col + h) * width_col + w;
        data_col[col_index] = im2col_get_pixel(data_im, height, width, im_row, im_col, c_im, l_pad, u_pad);
      }
    }
  }
}

void gemm(int M, int N, int K,
          scalar_t* A, int lda,
          scalar_t* B, int ldb,
          scalar_t* C, int ldc)
{
#ifdef OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, lda, B, ldb, 1, C, ldc);
#else
  int i,j,k;
  for (i = 0 ; i < M ; ++i) {
    for (k = 0 ; k < K ; ++k) {
      register scalar_t A_PART = A[i*lda + k];
      for (j = 0 ; j < N ; ++j) {
        C[i*ldc + j] += A_PART * B[k*ldb + j];
      }
    }
  }
#endif
}

void conv2d(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
            int out_ch, int in_ch, int k_size, int in_h, int in_w,
            int stride, int l_pad, int r_pad, int u_pad, int d_pad)
{
  int out_w = (in_w + l_pad+r_pad - k_size) / stride + 1;
  int out_h = (in_h + u_pad+d_pad - k_size) / stride + 1;
  
  scalar_t* data_col = (scalar_t*)calloc(in_ch*k_size*k_size*out_h*out_w, sizeof(scalar_t));

  im2col(INPUT, data_col, in_ch, in_h, in_w, k_size, stride, l_pad, r_pad, u_pad, d_pad);
  
  int M = out_ch, N = out_h*out_w, K = in_ch*k_size*k_size;
  gemm(M, N, K, WEIGHT, K, data_col, N, OUTPUT, N);
}
