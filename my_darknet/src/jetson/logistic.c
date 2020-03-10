#include <math.h>

#include "type.h"

inline scalar_t logistic_pixel(scalar_t x)
{
  return 1/(1+exp(-x));
}

void logistic(scalar_t* INPUT, scalar_t* OUTPUT, int size)
{
  int i;
  for (i = 0 ; i < size ; ++i) {
    OUTPUT[i] = logistic_pixel(INPUT[i]);
  }
}
