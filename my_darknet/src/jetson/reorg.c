#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "type.h"

void reorg(scalar_t* input, scalar_t* output, int stride, int ch, int h, int w)
{
  scalar_t* output_tmp = (scalar_t*)calloc(ch*h*w, sizeof(scalar_t));

  // The code below is copied from DarkNet blas.c/reorg_cpu
  int i,j,k;
  int out_c = ch / (stride*stride);

  for (k = 0 ; k < ch ; ++k) {
    for (j = 0 ; j < h ; ++j) {
      for (i = 0 ; i < w ; ++i) {
        int in_idx = i + w*(j + h*k);
        int c2 = k % out_c;
        int offset = k / out_c;
        int w2 = i*stride + offset % stride;
        int h2 = j*stride + offset / stride;
        int out_idx = w2 + w*stride*(h2 + h*stride*c2);
        output_tmp[in_idx] = input[out_idx];
      }
    }
  }

  memcpy(output, output_tmp, sizeof(scalar_t)*ch*h*w);
  free(output_tmp);
}
