#include "type.h"
#include "ops.h"
#include <stdio.h>
#include <stdlib.h>

void region(scalar_t* INPUT, float* OUTPUT,
            int n, int h, int w, int classes,
            int background, int coord, float threshold)
{
  int i,j,k;
  
  for (i = 0 ; i < n ; ++i) {
    logistic(INPUT+i*h*w*(classes+coord+!background), INPUT+i*h*w*(classes+coord+!background), 2*h*w);
    if (!background) {
      logistic(INPUT+i*h*w*(classes+coord+!background)+coord*h*w, INPUT+i*h*w*(classes+coord+!background)+coord*h*w, h*w);
    }
    for (j = 0 ; j < h*w ; ++j) {
      scalar_t* classes_list = (scalar_t*)calloc(classes, sizeof(scalar_t));
      int p;
      for (p = 0 ; p < classes ; ++p) {
        classes_list[p] = INPUT[i*h*w*(classes+coord+!background) + h*w*(coord+!background) + p*h*w + j];
      }
      softmax(classes_list, classes_list, classes, 1.0f);
      for (p = 0 ; p < classes ; ++p) {
        INPUT[i*h*w*(classes+coord+!background) + h*w*(coord+!background) + p*h*w + j] = classes_list[p];
      }
      free(classes_list);
    }
  }

  for (i = 0 ; i < h*w ; ++i) {
    for (j = 0 ; j < n ; ++j) {
      float scale = background ? 1 : INPUT[i + h*w*(coord + (classes+coord+1)*j)];
      if (scale > threshold) {
        for (k = 0 ; k < classes ; ++k) {
          float prob = scale * INPUT[i + h*w*(classes+coord+1)*j + h*w*(coord+1+k)];
          OUTPUT[k + classes*(i + h*w*j)] = prob*(prob > threshold);
        }
      }
    }
  }
}
