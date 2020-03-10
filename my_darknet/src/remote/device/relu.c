#include "ops.h"

inline scalar_t relu_pixel(scalar_t x)
{
  return (x > 0) * x;
}

void relu(scalar_t* INPUT, scalar_t* OUTPUT, int size)
{
  int i;
  for (i = 0 ; i < size ; ++i) {
    OUTPUT[i] = relu_pixel(INPUT[i]); 
  }
}
