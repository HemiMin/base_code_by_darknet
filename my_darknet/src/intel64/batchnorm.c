#include "batchnorm.h"

#include <math.h>
#include <stdio.h>

#include "ops.h"

void batchnorm(scalar_t* INPUT, scalar_t* OUTPUT, batch_ft* factors,
               int batch, int channel, int height, int width)
{
  int i,j,k;
  for (i = 0 ; i < batch ; ++i) {
    for (j = 0 ; j < channel ; ++j) {
      float scale = factors[j].scale;
      float mean = factors[j].mean; // rolling mean.
      float variance = factors[j].variance; // rolling variance
      float bias = factors[j].bias;
      for (k = 0 ; k < height*width ; ++k) {
        int index = k + height*width*(j + channel*i);
        OUTPUT[index] = scale * (INPUT[index]-mean) / (sqrt(variance) + .000001f) + bias;
      }
    }
  }
}
