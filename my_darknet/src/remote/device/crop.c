#include "crop.h"
#include "type.h"

#define SCALE 2
#define TRANS -1

inline scalar_t normalize(scalar_t x)
{
  return SCALE*x + TRANS;
}

image crop(image im, int w, int h)
{
  image out = make_image(w, h, im.c);
  int dh = (im.h - h) / 2;
  int dw = (im.w - w) / 2;
  int c,i,j;
  int row, col;
  int count = 0;
  for (c = 0 ; c < im.c ; ++c) {
    for (i = 0 ; i < h ; ++i) {
      row = i + dh;
      for (j = 0 ; j < w ; ++j) {
        col = j + dw;
        int index = col + im.w*(row + im.h*c);
        out.data[count++] = normalize(im.data[index]);
      }
    }
  }
  return out;
}
