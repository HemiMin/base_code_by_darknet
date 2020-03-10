#include "ops.h"

void gemv(int M, int N,
          scalar_t* vec,
          scalar_t* mat, int ldn,
          scalar_t* o_vec)
{
  int i,j;
  for (i = 0 ; i < M ; ++i) {
    register scalar_t o_val = 0;
    for (j = 0 ; j < N ; ++j) {
      o_val += vec[j] * mat[i*ldn + j];
    }
    o_vec[i] = o_val;
  }
}

void connected(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
                    int out_ch, int in_ch)
{
  gemv(out_ch, in_ch, INPUT, WEIGHT, in_ch, OUTPUT);
}
