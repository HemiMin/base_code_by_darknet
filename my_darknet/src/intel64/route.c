#include "ops.h"

#include <string.h>

void route_1(scalar_t* INPUT, scalar_t* OUTPUT, int batch, size_t size)
{
  memcpy(OUTPUT, INPUT, batch*size*sizeof(scalar_t));
}

void route_2(scalar_t* INPUT_1, scalar_t* INPUT_2, scalar_t* OUTPUT, int batch,
             size_t size_1, size_t size_2)
{
  int i;
  for (i = 0 ; i < batch ; ++i) {
    memcpy(OUTPUT+i*(size_1+size_2), INPUT_1+i*size_1, size_1*sizeof(scalar_t));
    memcpy(OUTPUT+i*(size_1+size_2)+size_1, INPUT_2+i*size_2, size_2*sizeof(scalar_t));
  }
}
