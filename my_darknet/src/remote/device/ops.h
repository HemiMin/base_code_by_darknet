#ifndef CONV_H_
#define CONV_H_

#include "type.h"

void conv2d(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
            int out_ch, int in_ch, int k_size, int in_h, int ih_w,
            int stride, int pad);

void addbias(scalar_t* INPUT, scalar_t* OUTPUT, scalar_t* BIASES,
             int ch, int size);

void addbias_connected(scalar_t* INPUT, scalar_t* OUTPUT, scalar_t* BIASES,
                       int size);

void relu(scalar_t* INPUT, scalar_t* OUTPUT, int size);

void maxpool2d(scalar_t* INPUT, scalar_t* OUTPUT,
               int ch, int h, int w, int k_size, int stride, int pad);

void connected(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
                    int out_ch, int in_ch);

void dropout(scalar_t* INPUT, scalar_t* OUTPUT, int size, float probability);

void softmax(scalar_t* INPUT, float* OUTPUT, int size, float temperature);

#endif
