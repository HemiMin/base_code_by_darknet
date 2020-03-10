#include "ops.h"

void addbias(scalar_t* INPUT, scalar_t* OUTPUT, scalar_t* BIASES, int ch, int size)
{
  int i,j;
  for (i = 0 ; i < ch ; ++i) {
    scalar_t bias = BIASES[i];
    for (j = 0 ; j < size ; ++j) {
      int index = j + size*i;
      OUTPUT[index] = INPUT[index] + bias;
    }
  }
}

void addbias_connected(scalar_t* INPUT, scalar_t* OUTPUT, scalar_t* BIASES, int size)
{
  int i;
  for (i = 0 ; i < size ; ++i) {
    OUTPUT[i] = INPUT[i] + BIASES[i];
  }
}
