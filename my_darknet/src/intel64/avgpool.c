#include "ops.h"

void avgpool2d( scalar_t* INPUT, scalar_t* OUTPUT,
                int ch, int h, int w)
{
  int i,j;
  scalar_t sum;

  for (i = 0 ; i < ch ; ++i) {
    sum = 0;
    for (j = 0 ; j < h*w ; ++j) {
      sum += INPUT[j + h*w*i];
    }
    OUTPUT[i] = sum / h / w;
  }
}
