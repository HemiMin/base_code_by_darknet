#include "ops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void route_1(scalar_t* INPUT, scalar_t* OUTPUT, int batch, size_t size)
{
  memcpy(OUTPUT, INPUT, batch*size*sizeof(scalar_t));
}

void route_2(scalar_t* INPUT_1, scalar_t* INPUT_2, scalar_t* OUTPUT, int batch,
             size_t size_1, size_t size_2)
{
  memcpy(OUTPUT, INPUT_1, batch*size_1*sizeof(scalar_t));
  memcpy(OUTPUT+batch*size_1, INPUT_2, batch*size_2*sizeof(scalar_t));
}
