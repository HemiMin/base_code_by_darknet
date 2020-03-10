#ifndef CONV_H_
#define CONV_H_

#include <stdlib.h>

#include "type.h"
#include "batchnorm.h"

void conv2d(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
            int out_ch, int in_ch, int k_size, int in_h, int ih_w,
            int stride, int l_pad, int r_pad, int u_pad, int d_pad);

#ifdef __cplusplus
extern "C" {
#endif
void conv2d_gpu(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
                int out_ch, int in_ch, int k_size, int in_h, int in_w,
                int stride, int l_pad, int r_pad, int u_pad, int d_pad);


void conv2d_gpu_except_memcpy(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
                              int out_ch, int in_ch, int k_size, int in_h, int in_w,
                              int stride, int l_pad, int r_pad, int u_pad, int d_pad);
void cudaMallocWrapper(void** ptr, size_t size);
void cudaFreeWrapper(scalar_t* ptr);
#ifdef __cplusplus
}
#endif

void batchnorm(scalar_t* INPUT, scalar_t* OUTPUT, batch_ft* factors,
               int batch, int channel, int height, int width);

void addbias(scalar_t* INPUT, scalar_t* OUTPUT, scalar_t* BIASES,
             int ch, int size);

void addbias_connected(scalar_t* INPUT, scalar_t* OUTPUT, scalar_t* BIASES,
                       int size);

void logistic(scalar_t* INPUT, scalar_t* OUTPUT, int size);
void relu(scalar_t* INPUT, scalar_t* OUTPUT, int size);
void leaky_relu(scalar_t* INPUT, scalar_t* OUTPUT, int size);

void maxpool2d(scalar_t* INPUT, scalar_t* OUTPUT,
               int ch, int h, int w, int k_size, int stride, int pad);
void avgpool2d(scalar_t* INPUT, scalar_t* OUTPUT, int ch, int h, int w);

void connected(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,
                    int out_ch, int in_ch);

void dropout(scalar_t* INPUT, scalar_t* OUTPUT, int size, float probability);

void residual(scalar_t* INPUT_1, scalar_t* INPUT_2, scalar_t* OUTPUT, int batch,
              int channel_1, int height_1, int width_1,
              int channel_2, int height_2, int width_2);
void route_1(scalar_t* INPUT, scalar_t* OUTPUT, int batch, size_t size);
void route_2( scalar_t* INPUT_1, scalar_t* INPUT_2, scalar_t* OUTPUT, int batch,
              size_t size_1, size_t size_2);

void reorg(scalar_t* input, scalar_t* output, int stride, int ch, int h, int w);

void softmax(scalar_t* INPUT, float* OUTPUT, int size, float temperature);
void region(scalar_t* INPUT, float* OUTPUT,
            int n, int h, int w, int classes,
            int background, int coord, float threshold);

#endif
