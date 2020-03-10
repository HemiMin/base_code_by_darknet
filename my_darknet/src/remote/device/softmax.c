#include <math.h>
#include <float.h>

#include "ops.h"

#define MAX_VAL FLT_MAX

void softmax(scalar_t* INPUT, float* OUTPUT, int size, float temperature)
{
  int i;
  scalar_t largest = -MAX_VAL;
  for (i = 0 ; i < size ; ++i) {
    scalar_t val = INPUT[i];
    if (val > largest) largest = val;
  }

  float sum = 0;
  for (i = 0 ; i < size ; ++i) {
    float e = exp(INPUT[i]/temperature - largest/temperature);
    sum += e;
    OUTPUT[i] = e;
  }
  for (i = 0 ; i < size ; ++i) {
    OUTPUT[i] /= sum;
  }
}
