#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ops.h"
#include "type.h"
#include "image.h"
#include "classifier.h"
#include "topk.h"
#include "crop.h"
#if defined(CUDA) || defined(CUBLAS)
#include "timer.h"
#endif

#define  TOPK  5
#define  IN_MEM_SIZE 376832000
#define  WT_MEM_SIZE 294912000
#define  OT_MEM_SIZE 147456000

#ifdef PLANNER
extern void conv_1_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_5_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_5_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_5_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
#endif

int main(int argc, char* argv[])
{
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <image file path>.jpg <weights file path>.weights <top k>", argv[0]);
    exit(0);
  }

  int top_K = atoi(argv[3]);

  // Image load
  image im = load_image(argv[1], 256, 256, 3);
  FILE* wt_fp = fopen(argv[2], "rb");
  if (wt_fp == NULL) {
    fprintf(stderr, "File %s is not opened.\n", argv[2]);
    exit(0);
  }

  // Memory Allocation
  scalar_t* wt_conv_1_1 = (scalar_t*)calloc(3*3*3*64, sizeof(scalar_t));
  scalar_t* bs_conv_1_1 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* ot_conv_1_1 = (scalar_t*)calloc(224*224*64, sizeof(scalar_t));
  scalar_t* wt_conv_1_2 = (scalar_t*)calloc(3*3*64*64, sizeof(scalar_t));
  scalar_t* bs_conv_1_2 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* ot_conv_1_2 = (scalar_t*)calloc(224*224*64, sizeof(scalar_t));
  scalar_t* ot_maxpol_1 = (scalar_t*)calloc(112*112*64, sizeof(scalar_t));
  scalar_t* wt_conv_2_1 = (scalar_t*)calloc(3*3*64*128, sizeof(scalar_t));
  scalar_t* bs_conv_2_1 = (scalar_t*)calloc(128, sizeof(scalar_t));
  scalar_t* ot_conv_2_1 = (scalar_t*)calloc(112*112*128, sizeof(scalar_t));
  scalar_t* wt_conv_2_2 = (scalar_t*)calloc(3*3*128*128, sizeof(scalar_t));
  scalar_t* bs_conv_2_2 = (scalar_t*)calloc(128, sizeof(scalar_t));
  scalar_t* ot_conv_2_2 = (scalar_t*)calloc(112*112*128, sizeof(scalar_t));
  scalar_t* ot_maxpol_2 = (scalar_t*)calloc(56*56*128, sizeof(scalar_t));
  scalar_t* wt_conv_3_1 = (scalar_t*)calloc(3*3*128*256, sizeof(scalar_t));
  scalar_t* bs_conv_3_1 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* ot_conv_3_1 = (scalar_t*)calloc(56*56*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_2 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  scalar_t* bs_conv_3_2 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* ot_conv_3_2 = (scalar_t*)calloc(56*56*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_3 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  scalar_t* bs_conv_3_3 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* ot_conv_3_3 = (scalar_t*)calloc(56*56*256, sizeof(scalar_t));
  scalar_t* ot_maxpol_3 = (scalar_t*)calloc(28*28*256, sizeof(scalar_t));
  scalar_t* wt_conv_4_1 = (scalar_t*)calloc(3*3*256*512, sizeof(scalar_t));
  scalar_t* bs_conv_4_1 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* ot_conv_4_1 = (scalar_t*)calloc(28*28*512, sizeof(scalar_t));
  scalar_t* wt_conv_4_2 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_4_2 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* ot_conv_4_2 = (scalar_t*)calloc(28*28*512, sizeof(scalar_t));
  scalar_t* wt_conv_4_3 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_4_3 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* ot_conv_4_3 = (scalar_t*)calloc(28*28*512, sizeof(scalar_t));
  scalar_t* ot_maxpol_4 = (scalar_t*)calloc(14*14*512, sizeof(scalar_t));
  scalar_t* wt_conv_5_1 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_5_1 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* ot_conv_5_1 = (scalar_t*)calloc(14*14*512, sizeof(scalar_t));
  scalar_t* wt_conv_5_2 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_5_2 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* ot_conv_5_2 = (scalar_t*)calloc(14*14*512, sizeof(scalar_t));
  scalar_t* wt_conv_5_3 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_5_3 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* ot_conv_5_3 = (scalar_t*)calloc(14*14*512, sizeof(scalar_t));
  scalar_t* ot_maxpol_5 = (scalar_t*)calloc(7*7*512, sizeof(scalar_t));
  scalar_t* wt_connct_1 = (scalar_t*)calloc(25088*4096, sizeof(scalar_t));
  scalar_t* bs_connct_1 = (scalar_t*)calloc(4096, sizeof(scalar_t));
  scalar_t* ot_connct_1 = (scalar_t*)calloc(1*1*4096, sizeof(scalar_t));
  scalar_t* wt_connct_2 = (scalar_t*)calloc(4096*4096, sizeof(scalar_t));
  scalar_t* bs_connct_2 = (scalar_t*)calloc(4096, sizeof(scalar_t));
  scalar_t* ot_connct_2 = (scalar_t*)calloc(1*1*4096, sizeof(scalar_t));
  scalar_t* wt_connct_3 = (scalar_t*)calloc(4096*1000, sizeof(scalar_t));
  scalar_t* bs_connct_3 = (scalar_t*)calloc(1000, sizeof(scalar_t));
  scalar_t* ot_connct_3 = (scalar_t*)calloc(1*1*1000, sizeof(scalar_t));
  float* ot_softmax = (float*)calloc(1000, sizeof(float));

  // Read weights and bias and batch normalization factors from file
  fread((char*)wt_conv_1_1, sizeof(scalar_t), 3*3*3*64, wt_fp);
  fread((char*)bs_conv_1_1, sizeof(scalar_t), 64, wt_fp);
  fread((char*)wt_conv_1_2, sizeof(scalar_t), 3*3*64*64, wt_fp);
  fread((char*)bs_conv_1_2, sizeof(scalar_t), 64, wt_fp);
  fread((char*)wt_conv_2_1, sizeof(scalar_t), 3*3*64*128, wt_fp);
  fread((char*)bs_conv_2_1, sizeof(scalar_t), 128, wt_fp);
  fread((char*)wt_conv_2_2, sizeof(scalar_t), 3*3*128*128, wt_fp);
  fread((char*)bs_conv_2_2, sizeof(scalar_t), 128, wt_fp);
  fread((char*)wt_conv_3_1, sizeof(scalar_t), 3*3*128*256, wt_fp);
  fread((char*)bs_conv_3_1, sizeof(scalar_t), 256, wt_fp);
  fread((char*)wt_conv_3_2, sizeof(scalar_t), 3*3*256*256, wt_fp);
  fread((char*)bs_conv_3_2, sizeof(scalar_t), 256, wt_fp);
  fread((char*)wt_conv_3_3, sizeof(scalar_t), 3*3*256*256, wt_fp);
  fread((char*)bs_conv_3_3, sizeof(scalar_t), 256, wt_fp);
  fread((char*)wt_conv_4_1, sizeof(scalar_t), 3*3*256*512, wt_fp);
  fread((char*)bs_conv_4_1, sizeof(scalar_t), 512, wt_fp);
  fread((char*)wt_conv_4_2, sizeof(scalar_t), 3*3*512*512, wt_fp);
  fread((char*)bs_conv_4_2, sizeof(scalar_t), 512, wt_fp);
  fread((char*)wt_conv_4_3, sizeof(scalar_t), 3*3*512*512, wt_fp);
  fread((char*)bs_conv_4_3, sizeof(scalar_t), 512, wt_fp);
  fread((char*)wt_conv_5_1, sizeof(scalar_t), 3*3*512*512, wt_fp);
  fread((char*)bs_conv_5_1, sizeof(scalar_t), 512, wt_fp);
  fread((char*)wt_conv_5_2, sizeof(scalar_t), 3*3*512*512, wt_fp);
  fread((char*)bs_conv_5_2, sizeof(scalar_t), 512, wt_fp);
  fread((char*)wt_conv_5_3, sizeof(scalar_t), 3*3*512*512, wt_fp);
  fread((char*)bs_conv_5_3, sizeof(scalar_t), 512, wt_fp);
  fread((char*)wt_connct_1, sizeof(scalar_t), 25088*4096, wt_fp);
  fread((char*)bs_connct_1, sizeof(scalar_t), 4096, wt_fp);
  fread((char*)wt_connct_2, sizeof(scalar_t), 4096*4096, wt_fp);
  fread((char*)bs_connct_2, sizeof(scalar_t), 4096, wt_fp);
  fread((char*)wt_connct_3, sizeof(scalar_t), 4096*1000, wt_fp);
  fread((char*)bs_connct_3, sizeof(scalar_t), 1000, wt_fp);

  fclose(wt_fp);

#ifdef PLANNER
  scalar_t* in_mem_0 = (scalar_t*)calloc(IN_MEM_SIZE, sizeof(scalar_t));
  scalar_t* wt_mem_0 = (scalar_t*)calloc(WT_MEM_SIZE, sizeof(scalar_t));
  scalar_t* ot_mem_0 = (scalar_t*)calloc(OT_MEM_SIZE, sizeof(scalar_t));
#endif

#if defined(CUDA) || defined(CUBLAS)
  init_timer();
#else
  clock_t start, end;

#endif

#ifdef TIME_ESTIMATE
  clock_t est_time;
#endif

  // Run Network
#if defined(CUDA) || defined(CUBLAS)
  start_timer();
#else
  start = clock();

#endif

  im = crop(im, 224, 224);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_1_1(im.data, wt_conv_1_1, ot_conv_1_1, 64, 3, 3, 224, 224, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(im.data, wt_conv_1_1, ot_conv_1_1, 64, 3, 3, 224, 224, 1, 1, 1, 1, 1);
#else
  conv2d(im.data, wt_conv_1_1, ot_conv_1_1, 64, 3, 3, 224, 224, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_1_1 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_1_1, ot_conv_1_1, bs_conv_1_1, 64, 224*224);
  relu(ot_conv_1_1, ot_conv_1_1, 1*64*224*224);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_1_2(ot_conv_1_1, wt_conv_1_2, ot_conv_1_2, 64, 64, 3, 224, 224, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_1_1, wt_conv_1_2, ot_conv_1_2, 64, 64, 3, 224, 224, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_1_1, wt_conv_1_2, ot_conv_1_2, 64, 64, 3, 224, 224, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_1_2 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_1_2, ot_conv_1_2, bs_conv_1_2, 64, 224*224);
  relu(ot_conv_1_2, ot_conv_1_2, 1*64*224*224);

  maxpool2d(ot_conv_1_2, ot_maxpol_1, 64, 224, 224, 2, 2, 1);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_2_1(ot_maxpol_1, wt_conv_2_1, ot_conv_2_1, 128, 64, 3, 112, 112, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_maxpol_1, wt_conv_2_1, ot_conv_2_1, 128, 64, 3, 112, 112, 1, 1, 1, 1, 1);
#else
  conv2d(ot_maxpol_1, wt_conv_2_1, ot_conv_2_1, 128, 64, 3, 112, 112, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_2_1 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_2_1, ot_conv_2_1, bs_conv_2_1, 128, 112*112);
  relu(ot_conv_2_1, ot_conv_2_1, 1*128*112*112);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_2_2(ot_conv_2_1, wt_conv_2_2, ot_conv_2_2, 128, 128, 3, 112, 112, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_2_1, wt_conv_2_2, ot_conv_2_2, 128, 128, 3, 112, 112, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_2_1, wt_conv_2_2, ot_conv_2_2, 128, 128, 3, 112, 112, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_2_2 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_2_2, ot_conv_2_2, bs_conv_2_2, 128, 112*112);
  relu(ot_conv_2_2, ot_conv_2_2, 1*128*112*112);

  maxpool2d(ot_conv_2_2, ot_maxpol_2, 128, 112, 112, 2, 2, 1);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_3_1(ot_maxpol_2, wt_conv_3_1, ot_conv_3_1, 256, 128, 3, 56, 56, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_maxpol_2, wt_conv_3_1, ot_conv_3_1, 256, 128, 3, 56, 56, 1, 1, 1, 1, 1);
#else
  conv2d(ot_maxpol_2, wt_conv_3_1, ot_conv_3_1, 256, 128, 3, 56, 56, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_3_1 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_3_1, ot_conv_3_1, bs_conv_3_1, 256, 56*56);
  relu(ot_conv_3_1, ot_conv_3_1, 1*256*56*56);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_3_2(ot_conv_3_1, wt_conv_3_2, ot_conv_3_2, 256, 256, 3, 56, 56, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_3_1, wt_conv_3_2, ot_conv_3_2, 256, 256, 3, 56, 56, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_3_1, wt_conv_3_2, ot_conv_3_2, 256, 256, 3, 56, 56, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_3_2 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_3_2, ot_conv_3_2, bs_conv_3_2, 256, 56*56);
  relu(ot_conv_3_2, ot_conv_3_2, 1*256*56*56);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_3_3(ot_conv_3_2, wt_conv_3_3, ot_conv_3_3, 256, 256, 3, 56, 56, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_3_2, wt_conv_3_3, ot_conv_3_3, 256, 256, 3, 56, 56, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_3_2, wt_conv_3_3, ot_conv_3_3, 256, 256, 3, 56, 56, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_3_3 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_3_3, ot_conv_3_3, bs_conv_3_3, 256, 56*56);
  relu(ot_conv_3_3, ot_conv_3_3, 1*256*56*56);

  maxpool2d(ot_conv_3_3, ot_maxpol_3, 256, 56, 56, 2, 2, 1);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_4_1(ot_maxpol_3, wt_conv_4_1, ot_conv_4_1, 512, 256, 3, 28, 28, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_maxpol_3, wt_conv_4_1, ot_conv_4_1, 512, 256, 3, 28, 28, 1, 1, 1, 1, 1);
#else
  conv2d(ot_maxpol_3, wt_conv_4_1, ot_conv_4_1, 512, 256, 3, 28, 28, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_4_1 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_4_1, ot_conv_4_1, bs_conv_4_1, 512, 28*28);
  relu(ot_conv_4_1, ot_conv_4_1, 1*512*28*28);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_4_2(ot_conv_4_1, wt_conv_4_2, ot_conv_4_2, 512, 512, 3, 28, 28, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_4_1, wt_conv_4_2, ot_conv_4_2, 512, 512, 3, 28, 28, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_4_1, wt_conv_4_2, ot_conv_4_2, 512, 512, 3, 28, 28, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_4_2 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_4_2, ot_conv_4_2, bs_conv_4_2, 512, 28*28);
  relu(ot_conv_4_2, ot_conv_4_2, 1*512*28*28);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_4_3(ot_conv_4_2, wt_conv_4_3, ot_conv_4_3, 512, 512, 3, 28, 28, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_4_2, wt_conv_4_3, ot_conv_4_3, 512, 512, 3, 28, 28, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_4_2, wt_conv_4_3, ot_conv_4_3, 512, 512, 3, 28, 28, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_4_3 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_4_3, ot_conv_4_3, bs_conv_4_3, 512, 28*28);
  relu(ot_conv_4_3, ot_conv_4_3, 1*512*28*28);

  maxpool2d(ot_conv_4_3, ot_maxpol_4, 512, 28, 28, 2, 2, 1);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_5_1(ot_maxpol_4, wt_conv_5_1, ot_conv_5_1, 512, 512, 3, 14, 14, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_maxpol_4, wt_conv_5_1, ot_conv_5_1, 512, 512, 3, 14, 14, 1, 1, 1, 1, 1);
#else
  conv2d(ot_maxpol_4, wt_conv_5_1, ot_conv_5_1, 512, 512, 3, 14, 14, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_5_1 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_5_1, ot_conv_5_1, bs_conv_5_1, 512, 14*14);
  relu(ot_conv_5_1, ot_conv_5_1, 1*512*14*14);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_5_2(ot_conv_5_1, wt_conv_5_2, ot_conv_5_2, 512, 512, 3, 14, 14, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_5_1, wt_conv_5_2, ot_conv_5_2, 512, 512, 3, 14, 14, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_5_1, wt_conv_5_2, ot_conv_5_2, 512, 512, 3, 14, 14, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_5_2 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_5_2, ot_conv_5_2, bs_conv_5_2, 512, 14*14);
  relu(ot_conv_5_2, ot_conv_5_2, 1*512*14*14);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_5_3(ot_conv_5_2, wt_conv_5_3, ot_conv_5_3, 512, 512, 3, 14, 14, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_5_2, wt_conv_5_3, ot_conv_5_3, 512, 512, 3, 14, 14, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_5_2, wt_conv_5_3, ot_conv_5_3, 512, 512, 3, 14, 14, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_5_3 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_5_3, ot_conv_5_3, bs_conv_5_3, 512, 14*14);
  relu(ot_conv_5_3, ot_conv_5_3, 1*512*14*14);

  maxpool2d(ot_conv_5_3, ot_maxpol_5, 512, 14, 14, 2, 2, 1);

  connected(ot_maxpol_5, wt_connct_1, ot_connct_1, 4096, 25088);
  addbias_connected(ot_connct_1, ot_connct_1, bs_connct_1, 4096);
  relu(ot_connct_1, ot_connct_1, 1*4096*1*1);

  connected(ot_connct_1, wt_connct_2, ot_connct_2, 4096, 4096);
  addbias_connected(ot_connct_2, ot_connct_2, bs_connct_2, 4096);
  relu(ot_connct_2, ot_connct_2, 1*4096*1*1);

  connected(ot_connct_2, wt_connct_3, ot_connct_3, 1000, 4096);
  addbias_connected(ot_connct_3, ot_connct_3, bs_connct_3, 1000);
  softmax(ot_connct_3, ot_softmax, 1000, 1.000000);

#if defined(CUDA) || defined(CUBLAS)
  float elapsed_time_ms;
  stop_timer(&elapsed_time_ms);
  printf("Elapsed Time (GPU): %.3f msec\n", elapsed_time_ms);
#else
  end = clock();
  printf("Elapsed Time (CPU): %.3f msec\n", (float)(end-start)/CLOCKS_PER_SEC*1000);
#endif

  int i;
  int topk_idx[top_K];
  top_k(ot_softmax, 1000, top_K, topk_idx);  // Extract indexes of top K possible results
  float topk_pos[top_K];  // Possibilities of top K results.
  for (i = 0 ; i < top_K ; ++i) {
    topk_pos[i] = ot_softmax[topk_idx[i]];
  }
  sort_top_k(topk_pos, topk_idx, top_K);  // Sort topk_pos and topk_idx.

  for (i = 0 ; i < top_K ; ++i) {
    int class_num = topk_idx[i];
    char label[STR_SIZE];
    get_darknet_label(label, class_num);  // Find labels corresponding to class number.
    printf("(%5.2f%%) %s\n", topk_pos[i]*100, label);
  }

  // Deallocate resources
  free(wt_conv_1_1);
  free(bs_conv_1_1);
  free(ot_conv_1_1);
  free(wt_conv_1_2);
  free(bs_conv_1_2);
  free(ot_conv_1_2);
  free(ot_maxpol_1);
  free(wt_conv_2_1);
  free(bs_conv_2_1);
  free(ot_conv_2_1);
  free(wt_conv_2_2);
  free(bs_conv_2_2);
  free(ot_conv_2_2);
  free(ot_maxpol_2);
  free(wt_conv_3_1);
  free(bs_conv_3_1);
  free(ot_conv_3_1);
  free(wt_conv_3_2);
  free(bs_conv_3_2);
  free(ot_conv_3_2);
  free(wt_conv_3_3);
  free(bs_conv_3_3);
  free(ot_conv_3_3);
  free(ot_maxpol_3);
  free(wt_conv_4_1);
  free(bs_conv_4_1);
  free(ot_conv_4_1);
  free(wt_conv_4_2);
  free(bs_conv_4_2);
  free(ot_conv_4_2);
  free(wt_conv_4_3);
  free(bs_conv_4_3);
  free(ot_conv_4_3);
  free(ot_maxpol_4);
  free(wt_conv_5_1);
  free(bs_conv_5_1);
  free(ot_conv_5_1);
  free(wt_conv_5_2);
  free(bs_conv_5_2);
  free(ot_conv_5_2);
  free(wt_conv_5_3);
  free(bs_conv_5_3);
  free(ot_conv_5_3);
  free(ot_maxpol_5);
  free(wt_connct_1);
  free(bs_connct_1);
  free(ot_connct_1);
  free(wt_connct_2);
  free(bs_connct_2);
  free(ot_connct_2);
  free(wt_connct_3);
  free(bs_connct_3);
  free(ot_connct_3);
  free(ot_softmax);

  return 0;
}
