#include <string.h>

#include "ops.h"

#include "type.h"

void residual(scalar_t* INPUT_1, scalar_t* INPUT_2, scalar_t* OUTPUT, int batch, 
              int channel_1, int height_1, int width_1,
              int channel_2, int height_2, int width_2)
{
  int stride = width_1/width_2;  // 0
  int sample = width_2/width_1;  // 4
  if (stride < 1) stride = 1;  // 1
  if (sample < 1) sample = 1;  // 4
  int minw = (width_1 < width_2) ? width_1 : width_2;  // 64
  int minh = (height_1 < height_2) ? height_1 : height_2;  // 64
  int minc = (channel_1 < channel_2) ? channel_1 : channel_2;  // 64

  memcpy(OUTPUT, INPUT_2, channel_2*height_2*width_2*sizeof(scalar_t));

  int i,j,k,b;
  for (b = 0 ; b < batch ; ++b) {  // 1
    for (k = 0 ; k < minc ; ++k) {  // 64
      for (j = 0 ; j < minh ; ++j) {  // 64
        for (i = 0 ; i < minw ; ++i) {  // 64
          int index_1 = i*stride + width_1*(j*stride + height_1*(k + channel_1*b));
          int index_2 = i*sample + width_2*(j*sample + height_2*(k + channel_2*b));
          int index_out = index_2;
          OUTPUT[index_out] = INPUT_1[index_1] + INPUT_2[index_2];
        }
      }
    }
  }
}
