#ifndef IMAGE_H_
#define IMAGE_H_

#include "type.h"

typedef struct IMAGE {
  int w;
  int h;
  int c;
  scalar_t* data;
} image;

image make_image(int w, int h, int c);

image load_image(char* filename, int w, int h, int c);
void store_image(char* filename, image im);

#endif
