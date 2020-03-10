#include "ops.h"

#include <stdlib.h>
#include <time.h>

float generate_random(void)
{
  srand(2222222);
  return (float)rand()/RAND_MAX;
}

void dropout(scalar_t* INPUT, scalar_t* OUTPUT, int size, float probability)
{
  int i;
  for (i = 0 ; i < size ; ++i) {
    OUTPUT[i] = (generate_random() < probability) * INPUT[i];
  }
}
