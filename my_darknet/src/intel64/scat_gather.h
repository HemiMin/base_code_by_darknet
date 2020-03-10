#ifndef SCAT_GATHER_H_
#define SCAT_GATHER_H_

#include <stdlib.h>

#include "type.h"

#define HOST_TO_DEVICE  0
#define DEVICE_TO_HOST  1

size_t tile_cpy4d( scalar_t* dst, scalar_t* src, int start_idx,
                   int ldn,int ldc,int ldh,int ldw, int n,int c,int h,int w,
                   int flag);

#endif
