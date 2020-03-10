#include <float.h>

#include "ops.h"

#define MAX_VAL FLT_MAX

void maxpool2d(scalar_t* INPUT, scalar_t* OUTPUT,
               int ch, int h, int w, int k_size, int stride, int pad)
{
  int i,j,k,n,m;
  int out_h = (h + pad - k_size) / stride + 1;
  int out_w = (w + pad - k_size) / stride + 1;

  int w_offset = -pad/2;
  int h_offset = -pad/2;

  for (k = 0 ; k < ch ; ++k) {
    for (i = 0 ; i < out_h ; ++i) {
      for (j = 0 ; j < out_w ; ++j) {
        scalar_t max = -MAX_VAL;
        int out_index = j + out_w*(i + out_h*k);
        for (n = 0 ; n < k_size ; ++n) {
          for (m = 0 ; m < k_size ; ++m) {
            int cur_h = h_offset + i*stride + n;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + w*cur_h + w*h*k;
            int valid = (cur_h >= 0 && cur_h < h &&
                         cur_w >= 0 && cur_w < w);
            scalar_t val = (valid != 0) ? INPUT[index] : -MAX_VAL;
            if (val > max) {
              max = val;
            }
          }
        }
        OUTPUT[out_index] = max;
      }
    }
  }
}
