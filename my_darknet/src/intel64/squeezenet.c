#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ops.h"
#include "type.h"
#include "image.h"
#include "classifier.h"
#include "topk.h"
#include "crop.h"

#define  TOPK  5
#define  IN_MEM_SIZE 376832000
#define  WT_MEM_SIZE 294912000
#define  OT_MEM_SIZE 147456000

#ifdef PLANNER
extern void conv_0( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void sqz_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_1_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_1_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void sqz_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_2_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_2_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void sqz_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_3_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_3_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void sqz_4( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_4_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_4_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void sqz_5( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_5_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_5_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void sqz_6( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_6_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_6_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void sqz_7( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_7_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_7_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void sqz_8( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_8_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void exp_8_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_c( scalar_t*, scalar_t*, scalar_t*,
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
  scalar_t* wt_conv_0 = (scalar_t*)calloc(3*3*3*64, sizeof(scalar_t));
  scalar_t* bs_conv_0 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* ot_conv_0 = (scalar_t*)calloc(113*113*64, sizeof(scalar_t));
  scalar_t* ot_maxpol_1 = (scalar_t*)calloc(57*57*64, sizeof(scalar_t));
  scalar_t* wt_sqz_1 = (scalar_t*)calloc(1*1*64*16, sizeof(scalar_t));
  scalar_t* bs_sqz_1 = (scalar_t*)calloc(16, sizeof(scalar_t));
  scalar_t* ot_sqz_1 = (scalar_t*)calloc(57*57*16, sizeof(scalar_t));
  scalar_t* wt_exp_1_1 = (scalar_t*)calloc(1*1*16*64, sizeof(scalar_t));
  scalar_t* bs_exp_1_1 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* ot_exp_1_1 = (scalar_t*)calloc(57*57*64, sizeof(scalar_t));
  scalar_t* ot_route_1_1 = (scalar_t*)calloc(1*51984, sizeof(scalar_t));
  scalar_t* wt_exp_1_3 = (scalar_t*)calloc(3*3*16*64, sizeof(scalar_t));
  scalar_t* bs_exp_1_3 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* ot_exp_1_3 = (scalar_t*)calloc(57*57*64, sizeof(scalar_t));
  scalar_t* ot_route_1_2 = (scalar_t*)calloc(1*(207936+207936), sizeof(scalar_t));
  scalar_t* wt_sqz_2 = (scalar_t*)calloc(1*1*128*16, sizeof(scalar_t));
  scalar_t* bs_sqz_2 = (scalar_t*)calloc(16, sizeof(scalar_t));
  scalar_t* ot_sqz_2 = (scalar_t*)calloc(57*57*16, sizeof(scalar_t));
  scalar_t* wt_exp_2_1 = (scalar_t*)calloc(1*1*16*64, sizeof(scalar_t));
  scalar_t* bs_exp_2_1 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* ot_exp_2_1 = (scalar_t*)calloc(57*57*64, sizeof(scalar_t));
  scalar_t* ot_route_2_1 = (scalar_t*)calloc(1*51984, sizeof(scalar_t));
  scalar_t* wt_exp_2_3 = (scalar_t*)calloc(3*3*16*64, sizeof(scalar_t));
  scalar_t* bs_exp_2_3 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* ot_exp_2_3 = (scalar_t*)calloc(57*57*64, sizeof(scalar_t));
  scalar_t* ot_route_2_2 = (scalar_t*)calloc(1*(207936+207936), sizeof(scalar_t));
  scalar_t* ot_maxpol_2 = (scalar_t*)calloc(29*29*128, sizeof(scalar_t));
  scalar_t* wt_sqz_3 = (scalar_t*)calloc(1*1*128*32, sizeof(scalar_t));
  scalar_t* bs_sqz_3 = (scalar_t*)calloc(32, sizeof(scalar_t));
  scalar_t* ot_sqz_3 = (scalar_t*)calloc(29*29*32, sizeof(scalar_t));
  scalar_t* wt_exp_3_1 = (scalar_t*)calloc(1*1*32*128, sizeof(scalar_t));
  scalar_t* bs_exp_3_1 = (scalar_t*)calloc(128, sizeof(scalar_t));
  scalar_t* ot_exp_3_1 = (scalar_t*)calloc(29*29*128, sizeof(scalar_t));
  scalar_t* ot_route_3_1 = (scalar_t*)calloc(1*26912, sizeof(scalar_t));
  scalar_t* wt_exp_3_3 = (scalar_t*)calloc(3*3*32*128, sizeof(scalar_t));
  scalar_t* bs_exp_3_3 = (scalar_t*)calloc(128, sizeof(scalar_t));
  scalar_t* ot_exp_3_3 = (scalar_t*)calloc(29*29*128, sizeof(scalar_t));
  scalar_t* ot_route_3_2 = (scalar_t*)calloc(1*(107648+107648), sizeof(scalar_t));
  scalar_t* wt_sqz_4 = (scalar_t*)calloc(1*1*256*32, sizeof(scalar_t));
  scalar_t* bs_sqz_4 = (scalar_t*)calloc(32, sizeof(scalar_t));
  scalar_t* ot_sqz_4 = (scalar_t*)calloc(29*29*32, sizeof(scalar_t));
  scalar_t* wt_exp_4_1 = (scalar_t*)calloc(1*1*32*128, sizeof(scalar_t));
  scalar_t* bs_exp_4_1 = (scalar_t*)calloc(128, sizeof(scalar_t));
  scalar_t* ot_exp_4_1 = (scalar_t*)calloc(29*29*128, sizeof(scalar_t));
  scalar_t* ot_route_4_1 = (scalar_t*)calloc(1*26912, sizeof(scalar_t));
  scalar_t* wt_exp_4_3 = (scalar_t*)calloc(3*3*32*128, sizeof(scalar_t));
  scalar_t* bs_exp_4_3 = (scalar_t*)calloc(128, sizeof(scalar_t));
  scalar_t* ot_exp_4_3 = (scalar_t*)calloc(29*29*128, sizeof(scalar_t));
  scalar_t* ot_route_4_2 = (scalar_t*)calloc(1*(107648+107648), sizeof(scalar_t));
  scalar_t* ot_maxpol_3 = (scalar_t*)calloc(15*15*256, sizeof(scalar_t));
  scalar_t* wt_sqz_5 = (scalar_t*)calloc(1*1*256*48, sizeof(scalar_t));
  scalar_t* bs_sqz_5 = (scalar_t*)calloc(48, sizeof(scalar_t));
  scalar_t* ot_sqz_5 = (scalar_t*)calloc(15*15*48, sizeof(scalar_t));
  scalar_t* wt_exp_5_1 = (scalar_t*)calloc(1*1*48*192, sizeof(scalar_t));
  scalar_t* bs_exp_5_1 = (scalar_t*)calloc(192, sizeof(scalar_t));
  scalar_t* ot_exp_5_1 = (scalar_t*)calloc(15*15*192, sizeof(scalar_t));
  scalar_t* ot_route_5_1 = (scalar_t*)calloc(1*10800, sizeof(scalar_t));
  scalar_t* wt_exp_5_3 = (scalar_t*)calloc(3*3*48*192, sizeof(scalar_t));
  scalar_t* bs_exp_5_3 = (scalar_t*)calloc(192, sizeof(scalar_t));
  scalar_t* ot_exp_5_3 = (scalar_t*)calloc(15*15*192, sizeof(scalar_t));
  scalar_t* ot_route_5_2 = (scalar_t*)calloc(1*(43200+43200), sizeof(scalar_t));
  scalar_t* wt_sqz_6 = (scalar_t*)calloc(1*1*384*48, sizeof(scalar_t));
  scalar_t* bs_sqz_6 = (scalar_t*)calloc(48, sizeof(scalar_t));
  scalar_t* ot_sqz_6 = (scalar_t*)calloc(15*15*48, sizeof(scalar_t));
  scalar_t* wt_exp_6_1 = (scalar_t*)calloc(1*1*48*192, sizeof(scalar_t));
  scalar_t* bs_exp_6_1 = (scalar_t*)calloc(192, sizeof(scalar_t));
  scalar_t* ot_exp_6_1 = (scalar_t*)calloc(15*15*192, sizeof(scalar_t));
  scalar_t* ot_route_6_1 = (scalar_t*)calloc(1*10800, sizeof(scalar_t));
  scalar_t* wt_exp_6_3 = (scalar_t*)calloc(3*3*48*192, sizeof(scalar_t));
  scalar_t* bs_exp_6_3 = (scalar_t*)calloc(192, sizeof(scalar_t));
  scalar_t* ot_exp_6_3 = (scalar_t*)calloc(15*15*192, sizeof(scalar_t));
  scalar_t* ot_route_6_2 = (scalar_t*)calloc(1*(43200+43200), sizeof(scalar_t));
  scalar_t* wt_sqz_7 = (scalar_t*)calloc(1*1*384*64, sizeof(scalar_t));
  scalar_t* bs_sqz_7 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* ot_sqz_7 = (scalar_t*)calloc(15*15*64, sizeof(scalar_t));
  scalar_t* wt_exp_7_1 = (scalar_t*)calloc(1*1*64*256, sizeof(scalar_t));
  scalar_t* bs_exp_7_1 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* ot_exp_7_1 = (scalar_t*)calloc(15*15*256, sizeof(scalar_t));
  scalar_t* ot_route_7_1 = (scalar_t*)calloc(1*14400, sizeof(scalar_t));
  scalar_t* wt_exp_7_3 = (scalar_t*)calloc(3*3*64*256, sizeof(scalar_t));
  scalar_t* bs_exp_7_3 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* ot_exp_7_3 = (scalar_t*)calloc(15*15*256, sizeof(scalar_t));
  scalar_t* ot_route_7_2 = (scalar_t*)calloc(1*(57600+57600), sizeof(scalar_t));
  scalar_t* wt_sqz_8 = (scalar_t*)calloc(1*1*512*64, sizeof(scalar_t));
  scalar_t* bs_sqz_8 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* ot_sqz_8 = (scalar_t*)calloc(15*15*64, sizeof(scalar_t));
  scalar_t* wt_exp_8_1 = (scalar_t*)calloc(1*1*64*256, sizeof(scalar_t));
  scalar_t* bs_exp_8_1 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* ot_exp_8_1 = (scalar_t*)calloc(15*15*256, sizeof(scalar_t));
  scalar_t* ot_route_8_1 = (scalar_t*)calloc(1*14400, sizeof(scalar_t));
  scalar_t* wt_exp_8_3 = (scalar_t*)calloc(3*3*64*256, sizeof(scalar_t));
  scalar_t* bs_exp_8_3 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* ot_exp_8_3 = (scalar_t*)calloc(15*15*256, sizeof(scalar_t));
  scalar_t* ot_route_8_2 = (scalar_t*)calloc(1*(57600+57600), sizeof(scalar_t));
  scalar_t* wt_conv_c = (scalar_t*)calloc(1*1*512*1000, sizeof(scalar_t));
  scalar_t* bs_conv_c = (scalar_t*)calloc(1000, sizeof(scalar_t));
  scalar_t* ot_conv_c = (scalar_t*)calloc(15*15*1000, sizeof(scalar_t));
  scalar_t* ot_avgpol_1 = (scalar_t*)calloc(1000, sizeof(scalar_t));
  float* ot_softmax = (float*)calloc(1000, sizeof(float));

  // Read weights and bias and batch normalization factors from file
  fread((char*)wt_conv_0, sizeof(scalar_t), 3*3*3*64, wt_fp);
  fread((char*)bs_conv_0, sizeof(scalar_t), 64, wt_fp);
  fread((char*)wt_sqz_1, sizeof(scalar_t), 1*1*64*16, wt_fp);
  fread((char*)bs_sqz_1, sizeof(scalar_t), 16, wt_fp);
  fread((char*)wt_exp_1_1, sizeof(scalar_t), 1*1*16*64, wt_fp);
  fread((char*)bs_exp_1_1, sizeof(scalar_t), 64, wt_fp);
  fread((char*)wt_exp_1_3, sizeof(scalar_t), 3*3*16*64, wt_fp);
  fread((char*)bs_exp_1_3, sizeof(scalar_t), 64, wt_fp);
  fread((char*)wt_sqz_2, sizeof(scalar_t), 1*1*128*16, wt_fp);
  fread((char*)bs_sqz_2, sizeof(scalar_t), 16, wt_fp);
  fread((char*)wt_exp_2_1, sizeof(scalar_t), 1*1*16*64, wt_fp);
  fread((char*)bs_exp_2_1, sizeof(scalar_t), 64, wt_fp);
  fread((char*)wt_exp_2_3, sizeof(scalar_t), 3*3*16*64, wt_fp);
  fread((char*)bs_exp_2_3, sizeof(scalar_t), 64, wt_fp);
  fread((char*)wt_sqz_3, sizeof(scalar_t), 1*1*128*32, wt_fp);
  fread((char*)bs_sqz_3, sizeof(scalar_t), 32, wt_fp);
  fread((char*)wt_exp_3_1, sizeof(scalar_t), 1*1*32*128, wt_fp);
  fread((char*)bs_exp_3_1, sizeof(scalar_t), 128, wt_fp);
  fread((char*)wt_exp_3_3, sizeof(scalar_t), 3*3*32*128, wt_fp);
  fread((char*)bs_exp_3_3, sizeof(scalar_t), 128, wt_fp);
  fread((char*)wt_sqz_4, sizeof(scalar_t), 1*1*256*32, wt_fp);
  fread((char*)bs_sqz_4, sizeof(scalar_t), 32, wt_fp);
  fread((char*)wt_exp_4_1, sizeof(scalar_t), 1*1*32*128, wt_fp);
  fread((char*)bs_exp_4_1, sizeof(scalar_t), 128, wt_fp);
  fread((char*)wt_exp_4_3, sizeof(scalar_t), 3*3*32*128, wt_fp);
  fread((char*)bs_exp_4_3, sizeof(scalar_t), 128, wt_fp);
  fread((char*)wt_sqz_5, sizeof(scalar_t), 1*1*256*48, wt_fp);
  fread((char*)bs_sqz_5, sizeof(scalar_t), 48, wt_fp);
  fread((char*)wt_exp_5_1, sizeof(scalar_t), 1*1*48*192, wt_fp);
  fread((char*)bs_exp_5_1, sizeof(scalar_t), 192, wt_fp);
  fread((char*)wt_exp_5_3, sizeof(scalar_t), 3*3*48*192, wt_fp);
  fread((char*)bs_exp_5_3, sizeof(scalar_t), 192, wt_fp);
  fread((char*)wt_sqz_6, sizeof(scalar_t), 1*1*384*48, wt_fp);
  fread((char*)bs_sqz_6, sizeof(scalar_t), 48, wt_fp);
  fread((char*)wt_exp_6_1, sizeof(scalar_t), 1*1*48*192, wt_fp);
  fread((char*)bs_exp_6_1, sizeof(scalar_t), 192, wt_fp);
  fread((char*)wt_exp_6_3, sizeof(scalar_t), 3*3*48*192, wt_fp);
  fread((char*)bs_exp_6_3, sizeof(scalar_t), 192, wt_fp);
  fread((char*)wt_sqz_7, sizeof(scalar_t), 1*1*384*64, wt_fp);
  fread((char*)bs_sqz_7, sizeof(scalar_t), 64, wt_fp);
  fread((char*)wt_exp_7_1, sizeof(scalar_t), 1*1*64*256, wt_fp);
  fread((char*)bs_exp_7_1, sizeof(scalar_t), 256, wt_fp);
  fread((char*)wt_exp_7_3, sizeof(scalar_t), 3*3*64*256, wt_fp);
  fread((char*)bs_exp_7_3, sizeof(scalar_t), 256, wt_fp);
  fread((char*)wt_sqz_8, sizeof(scalar_t), 1*1*512*64, wt_fp);
  fread((char*)bs_sqz_8, sizeof(scalar_t), 64, wt_fp);
  fread((char*)wt_exp_8_1, sizeof(scalar_t), 1*1*64*256, wt_fp);
  fread((char*)bs_exp_8_1, sizeof(scalar_t), 256, wt_fp);
  fread((char*)wt_exp_8_3, sizeof(scalar_t), 3*3*64*256, wt_fp);
  fread((char*)bs_exp_8_3, sizeof(scalar_t), 256, wt_fp);
  fread((char*)wt_conv_c, sizeof(scalar_t), 1*1*512*1000, wt_fp);
  fread((char*)bs_conv_c, sizeof(scalar_t), 1000, wt_fp);

  fclose(wt_fp);

#ifdef PLANNER
  scalar_t* in_mem_0 = (scalar_t*)calloc(IN_MEM_SIZE, sizeof(scalar_t));
  scalar_t* wt_mem_0 = (scalar_t*)calloc(WT_MEM_SIZE, sizeof(scalar_t));
  scalar_t* ot_mem_0 = (scalar_t*)calloc(OT_MEM_SIZE, sizeof(scalar_t));
#endif

  clock_t start, end;

  // Run Network
  start = clock();

  im = crop(im, 227, 227);

#ifdef PLANNER
  conv_0(im.data, wt_conv_0, ot_conv_0, 64, 3, 3, 227, 227, 2, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(im.data, wt_conv_0, ot_conv_0, 64, 3, 3, 227, 227, 2, 0, 0, 0, 0);
#endif
  printf("conv output: %f\n", ot_conv_0[0]);
  addbias(ot_conv_0, ot_conv_0, bs_conv_0, 64, 113*113);
  relu(ot_conv_0, ot_conv_0, 1*64*113*113);

  maxpool2d(ot_conv_0, ot_maxpol_1, 64, 113, 113, 3, 2, 2);

#ifdef PLANNER
  sqz_1(ot_maxpol_1, wt_sqz_1, ot_sqz_1, 16, 64, 1, 57, 57, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_maxpol_1, wt_sqz_1, ot_sqz_1, 16, 64, 1, 57, 57, 1, 0, 0, 0, 0);
#endif
  printf("conv output: %f\n", ot_sqz_1[0]);
  addbias(ot_sqz_1, ot_sqz_1, bs_sqz_1, 16, 57*57);
  relu(ot_sqz_1, ot_sqz_1, 1*16*57*57);

#ifdef PLANNER
  exp_1_1(ot_sqz_1, wt_exp_1_1, ot_exp_1_1, 64, 16, 1, 57, 57, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_sqz_1, wt_exp_1_1, ot_exp_1_1, 64, 16, 1, 57, 57, 1, 0, 0, 0, 0);
#endif
  addbias(ot_exp_1_1, ot_exp_1_1, bs_exp_1_1, 64, 57*57);
  relu(ot_exp_1_1, ot_exp_1_1, 1*64*57*57);

  route_1(ot_sqz_1, ot_route_1_1, 1, 51984);
#ifdef PLANNER
  exp_1_3(ot_route_1_1, wt_exp_1_3, ot_exp_1_3, 64, 16, 3, 57, 57, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_1_1, wt_exp_1_3, ot_exp_1_3, 64, 16, 3, 57, 57, 1, 1, 1, 1, 1);
#endif
  addbias(ot_exp_1_3, ot_exp_1_3, bs_exp_1_3, 64, 57*57);
  relu(ot_exp_1_3, ot_exp_1_3, 1*64*57*57);

  route_2(ot_exp_1_1, ot_exp_1_3, ot_route_1_2, 1, 207936, 207936);
#ifdef PLANNER
  sqz_2(ot_route_1_2, wt_sqz_2, ot_sqz_2, 16, 128, 1, 57, 57, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_1_2, wt_sqz_2, ot_sqz_2, 16, 128, 1, 57, 57, 1, 0, 0, 0, 0);
#endif
  addbias(ot_sqz_2, ot_sqz_2, bs_sqz_2, 16, 57*57);
  relu(ot_sqz_2, ot_sqz_2, 1*16*57*57);

#ifdef PLANNER
  exp_2_1(ot_sqz_2, wt_exp_2_1, ot_exp_2_1, 64, 16, 1, 57, 57, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_sqz_2, wt_exp_2_1, ot_exp_2_1, 64, 16, 1, 57, 57, 1, 0, 0, 0, 0);
#endif
  addbias(ot_exp_2_1, ot_exp_2_1, bs_exp_2_1, 64, 57*57);
  relu(ot_exp_2_1, ot_exp_2_1, 1*64*57*57);

  route_1(ot_sqz_2, ot_route_2_1, 1, 51984);
#ifdef PLANNER
  exp_2_3(ot_route_2_1, wt_exp_2_3, ot_exp_2_3, 64, 16, 3, 57, 57, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_2_1, wt_exp_2_3, ot_exp_2_3, 64, 16, 3, 57, 57, 1, 1, 1, 1, 1);
#endif
  addbias(ot_exp_2_3, ot_exp_2_3, bs_exp_2_3, 64, 57*57);
  relu(ot_exp_2_3, ot_exp_2_3, 1*64*57*57);

  route_2(ot_exp_2_1, ot_exp_2_3, ot_route_2_2, 1, 207936, 207936);
  maxpool2d(ot_route_2_2, ot_maxpol_2, 128, 57, 57, 3, 2, 2);

#ifdef PLANNER
  sqz_3(ot_maxpol_2, wt_sqz_3, ot_sqz_3, 32, 128, 1, 29, 29, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_maxpol_2, wt_sqz_3, ot_sqz_3, 32, 128, 1, 29, 29, 1, 0, 0, 0, 0);
#endif
  addbias(ot_sqz_3, ot_sqz_3, bs_sqz_3, 32, 29*29);
  relu(ot_sqz_3, ot_sqz_3, 1*32*29*29);

#ifdef PLANNER
  exp_3_1(ot_sqz_3, wt_exp_3_1, ot_exp_3_1, 128, 32, 1, 29, 29, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_sqz_3, wt_exp_3_1, ot_exp_3_1, 128, 32, 1, 29, 29, 1, 0, 0, 0, 0);
#endif
  addbias(ot_exp_3_1, ot_exp_3_1, bs_exp_3_1, 128, 29*29);
  relu(ot_exp_3_1, ot_exp_3_1, 1*128*29*29);

  route_1(ot_sqz_3, ot_route_3_1, 1, 26912);
#ifdef PLANNER
  exp_3_3(ot_route_3_1, wt_exp_3_3, ot_exp_3_3, 128, 32, 3, 29, 29, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_3_1, wt_exp_3_3, ot_exp_3_3, 128, 32, 3, 29, 29, 1, 1, 1, 1, 1);
#endif
  addbias(ot_exp_3_3, ot_exp_3_3, bs_exp_3_3, 128, 29*29);
  relu(ot_exp_3_3, ot_exp_3_3, 1*128*29*29);

  route_2(ot_exp_3_1, ot_exp_3_3, ot_route_3_2, 1, 107648, 107648);
#ifdef PLANNER
  sqz_4(ot_route_3_2, wt_sqz_4, ot_sqz_4, 32, 256, 1, 29, 29, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_3_2, wt_sqz_4, ot_sqz_4, 32, 256, 1, 29, 29, 1, 0, 0, 0, 0);
#endif
  addbias(ot_sqz_4, ot_sqz_4, bs_sqz_4, 32, 29*29);
  relu(ot_sqz_4, ot_sqz_4, 1*32*29*29);

#ifdef PLANNER
  exp_4_1(ot_sqz_4, wt_exp_4_1, ot_exp_4_1, 128, 32, 1, 29, 29, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_sqz_4, wt_exp_4_1, ot_exp_4_1, 128, 32, 1, 29, 29, 1, 0, 0, 0, 0);
#endif
  addbias(ot_exp_4_1, ot_exp_4_1, bs_exp_4_1, 128, 29*29);
  relu(ot_exp_4_1, ot_exp_4_1, 1*128*29*29);

  route_1(ot_sqz_4, ot_route_4_1, 1, 26912);
#ifdef PLANNER
  exp_4_3(ot_route_4_1, wt_exp_4_3, ot_exp_4_3, 128, 32, 3, 29, 29, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_4_1, wt_exp_4_3, ot_exp_4_3, 128, 32, 3, 29, 29, 1, 1, 1, 1, 1);
#endif
  addbias(ot_exp_4_3, ot_exp_4_3, bs_exp_4_3, 128, 29*29);
  relu(ot_exp_4_3, ot_exp_4_3, 1*128*29*29);

  route_2(ot_exp_4_1, ot_exp_4_3, ot_route_4_2, 1, 107648, 107648);
  maxpool2d(ot_route_4_2, ot_maxpol_3, 256, 29, 29, 3, 2, 2);

#ifdef PLANNER
  sqz_5(ot_maxpol_3, wt_sqz_5, ot_sqz_5, 48, 256, 1, 15, 15, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_maxpol_3, wt_sqz_5, ot_sqz_5, 48, 256, 1, 15, 15, 1, 0, 0, 0, 0);
#endif
  addbias(ot_sqz_5, ot_sqz_5, bs_sqz_5, 48, 15*15);
  relu(ot_sqz_5, ot_sqz_5, 1*48*15*15);

#ifdef PLANNER
  exp_5_1(ot_sqz_5, wt_exp_5_1, ot_exp_5_1, 192, 48, 1, 15, 15, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_sqz_5, wt_exp_5_1, ot_exp_5_1, 192, 48, 1, 15, 15, 1, 0, 0, 0, 0);
#endif
  addbias(ot_exp_5_1, ot_exp_5_1, bs_exp_5_1, 192, 15*15);
  relu(ot_exp_5_1, ot_exp_5_1, 1*192*15*15);

  route_1(ot_sqz_5, ot_route_5_1, 1, 10800);
#ifdef PLANNER
  exp_5_3(ot_route_5_1, wt_exp_5_3, ot_exp_5_3, 192, 48, 3, 15, 15, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_5_1, wt_exp_5_3, ot_exp_5_3, 192, 48, 3, 15, 15, 1, 1, 1, 1, 1);
#endif
  addbias(ot_exp_5_3, ot_exp_5_3, bs_exp_5_3, 192, 15*15);
  relu(ot_exp_5_3, ot_exp_5_3, 1*192*15*15);

  route_2(ot_exp_5_1, ot_exp_5_3, ot_route_5_2, 1, 43200, 43200);
#ifdef PLANNER
  sqz_6(ot_route_5_2, wt_sqz_6, ot_sqz_6, 48, 384, 1, 15, 15, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_5_2, wt_sqz_6, ot_sqz_6, 48, 384, 1, 15, 15, 1, 0, 0, 0, 0);
#endif
  addbias(ot_sqz_6, ot_sqz_6, bs_sqz_6, 48, 15*15);
  relu(ot_sqz_6, ot_sqz_6, 1*48*15*15);

#ifdef PLANNER
  exp_6_1(ot_sqz_6, wt_exp_6_1, ot_exp_6_1, 192, 48, 1, 15, 15, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_sqz_6, wt_exp_6_1, ot_exp_6_1, 192, 48, 1, 15, 15, 1, 0, 0, 0, 0);
#endif
  addbias(ot_exp_6_1, ot_exp_6_1, bs_exp_6_1, 192, 15*15);
  relu(ot_exp_6_1, ot_exp_6_1, 1*192*15*15);

  route_1(ot_sqz_6, ot_route_6_1, 1, 10800);
#ifdef PLANNER
  exp_6_3(ot_route_6_1, wt_exp_6_3, ot_exp_6_3, 192, 48, 3, 15, 15, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_6_1, wt_exp_6_3, ot_exp_6_3, 192, 48, 3, 15, 15, 1, 1, 1, 1, 1);
#endif
  addbias(ot_exp_6_3, ot_exp_6_3, bs_exp_6_3, 192, 15*15);
  relu(ot_exp_6_3, ot_exp_6_3, 1*192*15*15);

  route_2(ot_exp_6_1, ot_exp_6_3, ot_route_6_2, 1, 43200, 43200);
#ifdef PLANNER
  sqz_7(ot_route_6_2, wt_sqz_7, ot_sqz_7, 64, 384, 1, 15, 15, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_6_2, wt_sqz_7, ot_sqz_7, 64, 384, 1, 15, 15, 1, 0, 0, 0, 0);
#endif
  addbias(ot_sqz_7, ot_sqz_7, bs_sqz_7, 64, 15*15);
  relu(ot_sqz_7, ot_sqz_7, 1*64*15*15);

#ifdef PLANNER
  exp_7_1(ot_sqz_7, wt_exp_7_1, ot_exp_7_1, 256, 64, 1, 15, 15, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_sqz_7, wt_exp_7_1, ot_exp_7_1, 256, 64, 1, 15, 15, 1, 0, 0, 0, 0);
#endif
  addbias(ot_exp_7_1, ot_exp_7_1, bs_exp_7_1, 256, 15*15);
  relu(ot_exp_7_1, ot_exp_7_1, 1*256*15*15);

  route_1(ot_sqz_7, ot_route_7_1, 1, 14400);
#ifdef PLANNER
  exp_7_3(ot_route_7_1, wt_exp_7_3, ot_exp_7_3, 256, 64, 3, 15, 15, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_7_1, wt_exp_7_3, ot_exp_7_3, 256, 64, 3, 15, 15, 1, 1, 1, 1, 1);
#endif
  addbias(ot_exp_7_3, ot_exp_7_3, bs_exp_7_3, 256, 15*15);
  relu(ot_exp_7_3, ot_exp_7_3, 1*256*15*15);

  route_2(ot_exp_7_1, ot_exp_7_3, ot_route_7_2, 1, 57600, 57600);
#ifdef PLANNER
  sqz_8(ot_route_7_2, wt_sqz_8, ot_sqz_8, 64, 512, 1, 15, 15, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_7_2, wt_sqz_8, ot_sqz_8, 64, 512, 1, 15, 15, 1, 0, 0, 0, 0);
#endif
  addbias(ot_sqz_8, ot_sqz_8, bs_sqz_8, 64, 15*15);
  relu(ot_sqz_8, ot_sqz_8, 1*64*15*15);

#ifdef PLANNER
  exp_8_1(ot_sqz_8, wt_exp_8_1, ot_exp_8_1, 256, 64, 1, 15, 15, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_sqz_8, wt_exp_8_1, ot_exp_8_1, 256, 64, 1, 15, 15, 1, 0, 0, 0, 0);
#endif
  addbias(ot_exp_8_1, ot_exp_8_1, bs_exp_8_1, 256, 15*15);
  relu(ot_exp_8_1, ot_exp_8_1, 1*256*15*15);

  route_1(ot_sqz_8, ot_route_8_1, 1, 14400);
#ifdef PLANNER
  exp_8_3(ot_route_8_1, wt_exp_8_3, ot_exp_8_3, 256, 64, 3, 15, 15, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_8_1, wt_exp_8_3, ot_exp_8_3, 256, 64, 3, 15, 15, 1, 1, 1, 1, 1);
#endif
  addbias(ot_exp_8_3, ot_exp_8_3, bs_exp_8_3, 256, 15*15);
  relu(ot_exp_8_3, ot_exp_8_3, 1*256*15*15);

  route_2(ot_exp_8_1, ot_exp_8_3, ot_route_8_2, 1, 57600, 57600);
#ifdef PLANNER
  conv_c(ot_route_8_2, wt_conv_c, ot_conv_c, 1000, 512, 1, 15, 15, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_route_8_2, wt_conv_c, ot_conv_c, 1000, 512, 1, 15, 15, 1, 0, 0, 0, 0);
#endif
  addbias(ot_conv_c, ot_conv_c, bs_conv_c, 1000, 15*15);
  relu(ot_conv_c, ot_conv_c, 1*1000*15*15);

  avgpool2d(ot_conv_c, ot_avgpol_1, 1000, 15, 15);

  softmax(ot_avgpol_1, ot_softmax, 1000, 1.000000);

  end = clock();

  printf("Elapsed Time: %.3f msec\n", (float)(end-start)/CLOCKS_PER_SEC*1000);

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
  free(wt_conv_0);
  free(bs_conv_0);
  free(ot_conv_0);
  free(ot_maxpol_1);
  free(wt_sqz_1);
  free(bs_sqz_1);
  free(ot_sqz_1);
  free(wt_exp_1_1);
  free(bs_exp_1_1);
  free(ot_exp_1_1);
  free(ot_route_1_1);
  free(wt_exp_1_3);
  free(bs_exp_1_3);
  free(ot_exp_1_3);
  free(ot_route_1_2);
  free(wt_sqz_2);
  free(bs_sqz_2);
  free(ot_sqz_2);
  free(wt_exp_2_1);
  free(bs_exp_2_1);
  free(ot_exp_2_1);
  free(ot_route_2_1);
  free(wt_exp_2_3);
  free(bs_exp_2_3);
  free(ot_exp_2_3);
  free(ot_route_2_2);
  free(ot_maxpol_2);
  free(wt_sqz_3);
  free(bs_sqz_3);
  free(ot_sqz_3);
  free(wt_exp_3_1);
  free(bs_exp_3_1);
  free(ot_exp_3_1);
  free(ot_route_3_1);
  free(wt_exp_3_3);
  free(bs_exp_3_3);
  free(ot_exp_3_3);
  free(ot_route_3_2);
  free(wt_sqz_4);
  free(bs_sqz_4);
  free(ot_sqz_4);
  free(wt_exp_4_1);
  free(bs_exp_4_1);
  free(ot_exp_4_1);
  free(ot_route_4_1);
  free(wt_exp_4_3);
  free(bs_exp_4_3);
  free(ot_exp_4_3);
  free(ot_route_4_2);
  free(ot_maxpol_3);
  free(wt_sqz_5);
  free(bs_sqz_5);
  free(ot_sqz_5);
  free(wt_exp_5_1);
  free(bs_exp_5_1);
  free(ot_exp_5_1);
  free(ot_route_5_1);
  free(wt_exp_5_3);
  free(bs_exp_5_3);
  free(ot_exp_5_3);
  free(ot_route_5_2);
  free(wt_sqz_6);
  free(bs_sqz_6);
  free(ot_sqz_6);
  free(wt_exp_6_1);
  free(bs_exp_6_1);
  free(ot_exp_6_1);
  free(ot_route_6_1);
  free(wt_exp_6_3);
  free(bs_exp_6_3);
  free(ot_exp_6_3);
  free(ot_route_6_2);
  free(wt_sqz_7);
  free(bs_sqz_7);
  free(ot_sqz_7);
  free(wt_exp_7_1);
  free(bs_exp_7_1);
  free(ot_exp_7_1);
  free(ot_route_7_1);
  free(wt_exp_7_3);
  free(bs_exp_7_3);
  free(ot_exp_7_3);
  free(ot_route_7_2);
  free(wt_sqz_8);
  free(bs_sqz_8);
  free(ot_sqz_8);
  free(wt_exp_8_1);
  free(bs_exp_8_1);
  free(ot_exp_8_1);
  free(ot_route_8_1);
  free(wt_exp_8_3);
  free(bs_exp_8_3);
  free(ot_exp_8_3);
  free(ot_route_8_2);
  free(wt_conv_c);
  free(bs_conv_c);
  free(ot_conv_c);
  free(ot_avgpol_1);
  free(ot_softmax);

  return 0;
}
