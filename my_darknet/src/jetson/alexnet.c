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
#define  IN_MEM_SIZE 524288
#define  WT_MEM_SIZE 262144
#define  OT_MEM_SIZE 524288

#ifdef PLANNER
extern void conv_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_5( scalar_t*, scalar_t*, scalar_t*,
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
  image im = load_image(argv[1], 227, 227, 3);
  FILE* wt_fp = fopen(argv[2], "rb");
  if (wt_fp == NULL) {
    fprintf(stderr, "File %s is not opened.\n", argv[2]);
    exit(0);
  }

  // Memory Allocation
  scalar_t* wt_conv_1 = (scalar_t*)calloc(11*11*3*96, sizeof(scalar_t));
  scalar_t* bs_conv_1 = (scalar_t*)calloc(96, sizeof(scalar_t));
  scalar_t* ot_conv_1 = (scalar_t*)calloc(55*55*96, sizeof(scalar_t));
  scalar_t* ot_maxpol_1 = (scalar_t*)calloc(27*27*96, sizeof(scalar_t));
  scalar_t* wt_conv_2 = (scalar_t*)calloc(5*5*96*256, sizeof(scalar_t));
  scalar_t* bs_conv_2 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* ot_conv_2 = (scalar_t*)calloc(27*27*256, sizeof(scalar_t));
  scalar_t* ot_maxpol_2 = (scalar_t*)calloc(13*13*256, sizeof(scalar_t));
  scalar_t* wt_conv_3 = (scalar_t*)calloc(3*3*256*384, sizeof(scalar_t));
  scalar_t* bs_conv_3 = (scalar_t*)calloc(384, sizeof(scalar_t));
  scalar_t* ot_conv_3 = (scalar_t*)calloc(13*13*384, sizeof(scalar_t));
  scalar_t* wt_conv_4 = (scalar_t*)calloc(3*3*384*384, sizeof(scalar_t));
  scalar_t* bs_conv_4 = (scalar_t*)calloc(384, sizeof(scalar_t));
  scalar_t* ot_conv_4 = (scalar_t*)calloc(13*13*384, sizeof(scalar_t));
  scalar_t* wt_conv_5 = (scalar_t*)calloc(3*3*384*256, sizeof(scalar_t));
  scalar_t* bs_conv_5 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* ot_conv_5 = (scalar_t*)calloc(13*13*256, sizeof(scalar_t));
  scalar_t* ot_maxpol_3 = (scalar_t*)calloc(6*6*256, sizeof(scalar_t));
  scalar_t* wt_connct_1 = (scalar_t*)calloc(9216*4096, sizeof(scalar_t));
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
  fread((char*)wt_conv_1, sizeof(scalar_t), 11*11*3*96, wt_fp);
  fread((char*)bs_conv_1, sizeof(scalar_t), 96, wt_fp);
  fread((char*)wt_conv_2, sizeof(scalar_t), 5*5*96*256, wt_fp);
  fread((char*)bs_conv_2, sizeof(scalar_t), 256, wt_fp);
  fread((char*)wt_conv_3, sizeof(scalar_t), 3*3*256*384, wt_fp);
  fread((char*)bs_conv_3, sizeof(scalar_t), 384, wt_fp);
  fread((char*)wt_conv_4, sizeof(scalar_t), 3*3*384*384, wt_fp);
  fread((char*)bs_conv_4, sizeof(scalar_t), 384, wt_fp);
  fread((char*)wt_conv_5, sizeof(scalar_t), 3*3*384*256, wt_fp);
  fread((char*)bs_conv_5, sizeof(scalar_t), 256, wt_fp);
  fread((char*)wt_connct_1, sizeof(scalar_t), 9216*4096, wt_fp);
  fread((char*)bs_connct_1, sizeof(scalar_t), 4096, wt_fp);
  fread((char*)wt_connct_2, sizeof(scalar_t), 4096*4096, wt_fp);
  fread((char*)bs_connct_2, sizeof(scalar_t), 4096, wt_fp);
  fread((char*)wt_connct_3, sizeof(scalar_t), 4096*1000, wt_fp);
  fread((char*)bs_connct_3, sizeof(scalar_t), 1000, wt_fp);

  fclose(wt_fp);

#ifdef PLANNER
#if defined(CUDA) || defined(CUBLAS)
  scalar_t *in_mem_0=NULL, *wt_mem_0=NULL, *ot_mem_0=NULL;
  cudaMallocWrapper((void**)&in_mem_0, sizeof(scalar_t)*IN_MEM_SIZE);
  cudaMallocWrapper((void**)&wt_mem_0, sizeof(scalar_t)*WT_MEM_SIZE);
  cudaMallocWrapper((void**)&ot_mem_0, sizeof(scalar_t)*OT_MEM_SIZE);
#else
  scalar_t* in_mem_0 = (scalar_t*)calloc(IN_MEM_SIZE, sizeof(scalar_t));
  scalar_t* wt_mem_0 = (scalar_t*)calloc(WT_MEM_SIZE, sizeof(scalar_t));
  scalar_t* ot_mem_0 = (scalar_t*)calloc(OT_MEM_SIZE, sizeof(scalar_t));
#endif
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

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_1(im.data, wt_conv_1, ot_conv_1, 96, 3, 11, 227, 227, 4, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(im.data, wt_conv_1, ot_conv_1, 96, 3, 11, 227, 227, 4, 0, 0, 0, 0);
#else
  conv2d(im.data, wt_conv_1, ot_conv_1, 96, 3, 11, 227, 227, 4, 0, 0, 0, 0);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_1 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_1, ot_conv_1, bs_conv_1, 96, 55*55);
  relu(ot_conv_1, ot_conv_1, 1*96*55*55);

  maxpool2d(ot_conv_1, ot_maxpol_1, 96, 55, 55, 3, 2, 0);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_2(ot_maxpol_1, wt_conv_2, ot_conv_2, 256, 96, 5, 27, 27, 1, 2, 2, 2, 2, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_maxpol_1, wt_conv_2, ot_conv_2, 256, 96, 5, 27, 27, 1, 2, 2, 2, 2);
#else
  conv2d(ot_maxpol_1, wt_conv_2, ot_conv_2, 256, 96, 5, 27, 27, 1, 2, 2, 2, 2);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_2 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_2, ot_conv_2, bs_conv_2, 256, 27*27);
  relu(ot_conv_2, ot_conv_2, 1*256*27*27);

  maxpool2d(ot_conv_2, ot_maxpol_2, 256, 27, 27, 3, 2, 0);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_3(ot_maxpol_2, wt_conv_3, ot_conv_3, 384, 256, 3, 13, 13, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_maxpol_2, wt_conv_3, ot_conv_3, 384, 256, 3, 13, 13, 1, 1, 1, 1, 1);
#else
  conv2d(ot_maxpol_2, wt_conv_3, ot_conv_3, 384, 256, 3, 13, 13, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_3 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_3, ot_conv_3, bs_conv_3, 384, 13*13);
  relu(ot_conv_3, ot_conv_3, 1*384*13*13);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_4(ot_conv_3, wt_conv_4, ot_conv_4, 384, 384, 3, 13, 13, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_3, wt_conv_4, ot_conv_4, 384, 384, 3, 13, 13, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_3, wt_conv_4, ot_conv_4, 384, 384, 3, 13, 13, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_4 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_4, ot_conv_4, bs_conv_4, 384, 13*13);
  relu(ot_conv_4, ot_conv_4, 1*384*13*13);

#ifdef TIME_ESTIMATE
  est_time = clock();
#endif
#ifdef PLANNER
  conv_5(ot_conv_4, wt_conv_5, ot_conv_5, 256, 384, 3, 13, 13, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#elif defined(CUDA) || defined(CUBLAS)
  conv2d_gpu(ot_conv_4, wt_conv_5, ot_conv_5, 256, 384, 3, 13, 13, 1, 1, 1, 1, 1);
#else
  conv2d(ot_conv_4, wt_conv_5, ot_conv_5, 256, 384, 3, 13, 13, 1, 1, 1, 1, 1);
#endif
#ifdef TIME_ESTIMATE
  printf("conv_5 elapsed time: %.3f msec\n", (float)(clock()-est_time)/CLOCKS_PER_SEC*1000);
#endif
  addbias(ot_conv_5, ot_conv_5, bs_conv_5, 256, 13*13);
  relu(ot_conv_5, ot_conv_5, 1*256*13*13);

  maxpool2d(ot_conv_5, ot_maxpol_3, 256, 13, 13, 3, 2, 0);

  connected(ot_maxpol_3, wt_connct_1, ot_connct_1, 4096, 9216);
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
#ifdef PLANNER
#if defined(CUDA) || defined(CUBLAS)
  cudaFreeWrapper(in_mem_0);
  cudaFreeWrapper(wt_mem_0);
  cudaFreeWrapper(ot_mem_0);
#else
  free(in_mem_0);
  free(wt_mem_0);
  free(ot_mem_0);
#endif
#endif
  free(wt_conv_1);
  free(bs_conv_1);
  free(ot_conv_1);
  free(ot_maxpol_1);
  free(wt_conv_2);
  free(bs_conv_2);
  free(ot_conv_2);
  free(ot_maxpol_2);
  free(wt_conv_3);
  free(bs_conv_3);
  free(ot_conv_3);
  free(wt_conv_4);
  free(bs_conv_4);
  free(ot_conv_4);
  free(wt_conv_5);
  free(bs_conv_5);
  free(ot_conv_5);
  free(ot_maxpol_3);
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
