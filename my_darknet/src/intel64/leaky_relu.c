#include "ops.h"

inline scalar_t leaky_relu_pixel(scalar_t x)
{
  return (x > 0) ? x : .1*x;
}

void leaky_relu(scalar_t* INPUT, scalar_t* OUTPUT, int size)
{
  int i;
  for (i = 0 ; i < size ; ++i) {
    OUTPUT[i] = leaky_relu_pixel(INPUT[i]);
  }
}
