#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

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
extern void conv_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_0_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_0_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_0_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_1_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_1_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_1_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_2_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_2_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_1_2_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_0_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_0_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_0_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_1_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_1_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_1_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_2_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_2_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_2_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_3_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_3_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_2_3_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_0_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_0_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_0_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_1_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_1_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_1_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_2_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_2_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_2_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_3_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_3_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_3_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_4_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_4_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_4_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_5_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_5_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_3_5_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_0_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_0_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_0_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_1_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_1_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_1_3( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_2_1( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_2_2( scalar_t*, scalar_t*, scalar_t*,
                int, int, int, int, int,
                int, int, int, int, int,
                scalar_t*, scalar_t*, scalar_t*);
extern void conv_4_2_3( scalar_t*, scalar_t*, scalar_t*,
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
  scalar_t* wt_conv_1 = (scalar_t*)calloc(7*7*3*64, sizeof(scalar_t));
  batch_ft* bt_conv_1 = (batch_ft*)calloc(64, sizeof(batch_ft));
  scalar_t* ot_conv_1 = (scalar_t*)calloc(128*128*64, sizeof(scalar_t));
  scalar_t* ot_maxpol_1 = (scalar_t*)calloc(64*64*64, sizeof(scalar_t));
  scalar_t* wt_conv_1_0_1 = (scalar_t*)calloc(1*1*64*64, sizeof(scalar_t));
  batch_ft* bt_conv_1_0_1 = (batch_ft*)calloc(64, sizeof(batch_ft));
  scalar_t* ot_conv_1_0_1 = (scalar_t*)calloc(64*64*64, sizeof(scalar_t));
  scalar_t* wt_conv_1_0_2 = (scalar_t*)calloc(3*3*64*64, sizeof(scalar_t));
  batch_ft* bt_conv_1_0_2 = (batch_ft*)calloc(64, sizeof(batch_ft));
  scalar_t* ot_conv_1_0_2 = (scalar_t*)calloc(64*64*64, sizeof(scalar_t));
  scalar_t* wt_conv_1_0_3 = (scalar_t*)calloc(1*1*64*256, sizeof(scalar_t));
  batch_ft* bt_conv_1_0_3 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_1_0_3 = (scalar_t*)calloc(64*64*256, sizeof(scalar_t));
  scalar_t* ot_residual_1 = (scalar_t*)calloc(64*64*256, sizeof(scalar_t));
  scalar_t* wt_conv_1_1_1 = (scalar_t*)calloc(1*1*256*64, sizeof(scalar_t));
  batch_ft* bt_conv_1_1_1 = (batch_ft*)calloc(64, sizeof(batch_ft));
  scalar_t* ot_conv_1_1_1 = (scalar_t*)calloc(64*64*64, sizeof(scalar_t));
  scalar_t* wt_conv_1_1_2 = (scalar_t*)calloc(3*3*64*64, sizeof(scalar_t));
  batch_ft* bt_conv_1_1_2 = (batch_ft*)calloc(64, sizeof(batch_ft));
  scalar_t* ot_conv_1_1_2 = (scalar_t*)calloc(64*64*64, sizeof(scalar_t));
  scalar_t* wt_conv_1_1_3 = (scalar_t*)calloc(1*1*64*256, sizeof(scalar_t));
  batch_ft* bt_conv_1_1_3 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_1_1_3 = (scalar_t*)calloc(64*64*256, sizeof(scalar_t));
  scalar_t* ot_residual_2 = (scalar_t*)calloc(64*64*256, sizeof(scalar_t));
  scalar_t* wt_conv_1_2_1 = (scalar_t*)calloc(1*1*256*64, sizeof(scalar_t));
  batch_ft* bt_conv_1_2_1 = (batch_ft*)calloc(64, sizeof(batch_ft));
  scalar_t* ot_conv_1_2_1 = (scalar_t*)calloc(64*64*64, sizeof(scalar_t));
  scalar_t* wt_conv_1_2_2 = (scalar_t*)calloc(3*3*64*64, sizeof(scalar_t));
  batch_ft* bt_conv_1_2_2 = (batch_ft*)calloc(64, sizeof(batch_ft));
  scalar_t* ot_conv_1_2_2 = (scalar_t*)calloc(64*64*64, sizeof(scalar_t));
  scalar_t* wt_conv_1_2_3 = (scalar_t*)calloc(1*1*64*256, sizeof(scalar_t));
  batch_ft* bt_conv_1_2_3 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_1_2_3 = (scalar_t*)calloc(64*64*256, sizeof(scalar_t));
  scalar_t* ot_residual_3 = (scalar_t*)calloc(64*64*256, sizeof(scalar_t));
  scalar_t* wt_conv_2_0_1 = (scalar_t*)calloc(1*1*256*128, sizeof(scalar_t));
  batch_ft* bt_conv_2_0_1 = (batch_ft*)calloc(128, sizeof(batch_ft));
  scalar_t* ot_conv_2_0_1 = (scalar_t*)calloc(64*64*128, sizeof(scalar_t));
  scalar_t* wt_conv_2_0_2 = (scalar_t*)calloc(3*3*128*128, sizeof(scalar_t));
  batch_ft* bt_conv_2_0_2 = (batch_ft*)calloc(128, sizeof(batch_ft));
  scalar_t* ot_conv_2_0_2 = (scalar_t*)calloc(32*32*128, sizeof(scalar_t));
  scalar_t* wt_conv_2_0_3 = (scalar_t*)calloc(1*1*128*512, sizeof(scalar_t));
  batch_ft* bt_conv_2_0_3 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_2_0_3 = (scalar_t*)calloc(32*32*512, sizeof(scalar_t));
  scalar_t* ot_residual_4 = (scalar_t*)calloc(32*32*512, sizeof(scalar_t));
  scalar_t* wt_conv_2_1_1 = (scalar_t*)calloc(1*1*512*128, sizeof(scalar_t));
  batch_ft* bt_conv_2_1_1 = (batch_ft*)calloc(128, sizeof(batch_ft));
  scalar_t* ot_conv_2_1_1 = (scalar_t*)calloc(32*32*128, sizeof(scalar_t));
  scalar_t* wt_conv_2_1_2 = (scalar_t*)calloc(3*3*128*128, sizeof(scalar_t));
  batch_ft* bt_conv_2_1_2 = (batch_ft*)calloc(128, sizeof(batch_ft));
  scalar_t* ot_conv_2_1_2 = (scalar_t*)calloc(32*32*128, sizeof(scalar_t));
  scalar_t* wt_conv_2_1_3 = (scalar_t*)calloc(1*1*128*512, sizeof(scalar_t));
  batch_ft* bt_conv_2_1_3 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_2_1_3 = (scalar_t*)calloc(32*32*512, sizeof(scalar_t));
  scalar_t* ot_residual_5 = (scalar_t*)calloc(32*32*512, sizeof(scalar_t));
  scalar_t* wt_conv_2_2_1 = (scalar_t*)calloc(1*1*512*128, sizeof(scalar_t));
  batch_ft* bt_conv_2_2_1 = (batch_ft*)calloc(128, sizeof(batch_ft));
  scalar_t* ot_conv_2_2_1 = (scalar_t*)calloc(32*32*128, sizeof(scalar_t));
  scalar_t* wt_conv_2_2_2 = (scalar_t*)calloc(3*3*128*128, sizeof(scalar_t));
  batch_ft* bt_conv_2_2_2 = (batch_ft*)calloc(128, sizeof(batch_ft));
  scalar_t* ot_conv_2_2_2 = (scalar_t*)calloc(32*32*128, sizeof(scalar_t));
  scalar_t* wt_conv_2_2_3 = (scalar_t*)calloc(1*1*128*512, sizeof(scalar_t));
  batch_ft* bt_conv_2_2_3 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_2_2_3 = (scalar_t*)calloc(32*32*512, sizeof(scalar_t));
  scalar_t* ot_residual_6 = (scalar_t*)calloc(32*32*512, sizeof(scalar_t));
  scalar_t* wt_conv_2_3_1 = (scalar_t*)calloc(1*1*512*128, sizeof(scalar_t));
  batch_ft* bt_conv_2_3_1 = (batch_ft*)calloc(128, sizeof(batch_ft));
  scalar_t* ot_conv_2_3_1 = (scalar_t*)calloc(32*32*128, sizeof(scalar_t));
  scalar_t* wt_conv_2_3_2 = (scalar_t*)calloc(3*3*128*128, sizeof(scalar_t));
  batch_ft* bt_conv_2_3_2 = (batch_ft*)calloc(128, sizeof(batch_ft));
  scalar_t* ot_conv_2_3_2 = (scalar_t*)calloc(32*32*128, sizeof(scalar_t));
  scalar_t* wt_conv_2_3_3 = (scalar_t*)calloc(1*1*128*512, sizeof(scalar_t));
  batch_ft* bt_conv_2_3_3 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_2_3_3 = (scalar_t*)calloc(32*32*512, sizeof(scalar_t));
  scalar_t* ot_residual_7 = (scalar_t*)calloc(32*32*512, sizeof(scalar_t));
  scalar_t* wt_conv_3_0_1 = (scalar_t*)calloc(1*1*512*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_0_1 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_0_1 = (scalar_t*)calloc(32*32*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_0_2 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_0_2 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_0_2 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_0_3 = (scalar_t*)calloc(1*1*256*1024, sizeof(scalar_t));
  batch_ft* bt_conv_3_0_3 = (batch_ft*)calloc(1024, sizeof(batch_ft));
  scalar_t* ot_conv_3_0_3 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* ot_residual_8 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* wt_conv_3_1_1 = (scalar_t*)calloc(1*1*1024*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_1_1 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_1_1 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_1_2 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_1_2 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_1_2 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_1_3 = (scalar_t*)calloc(1*1*256*1024, sizeof(scalar_t));
  batch_ft* bt_conv_3_1_3 = (batch_ft*)calloc(1024, sizeof(batch_ft));
  scalar_t* ot_conv_3_1_3 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* ot_residual_9 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* wt_conv_3_2_1 = (scalar_t*)calloc(1*1*1024*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_2_1 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_2_1 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_2_2 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_2_2 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_2_2 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_2_3 = (scalar_t*)calloc(1*1*256*1024, sizeof(scalar_t));
  batch_ft* bt_conv_3_2_3 = (batch_ft*)calloc(1024, sizeof(batch_ft));
  scalar_t* ot_conv_3_2_3 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* ot_residual_10 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* wt_conv_3_3_1 = (scalar_t*)calloc(1*1*1024*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_3_1 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_3_1 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_3_2 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_3_2 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_3_2 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_3_3 = (scalar_t*)calloc(1*1*256*1024, sizeof(scalar_t));
  batch_ft* bt_conv_3_3_3 = (batch_ft*)calloc(1024, sizeof(batch_ft));
  scalar_t* ot_conv_3_3_3 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* ot_residual_11 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* wt_conv_3_4_1 = (scalar_t*)calloc(1*1*1024*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_4_1 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_4_1 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_4_2 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_4_2 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_4_2 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_4_3 = (scalar_t*)calloc(1*1*256*1024, sizeof(scalar_t));
  batch_ft* bt_conv_3_4_3 = (batch_ft*)calloc(1024, sizeof(batch_ft));
  scalar_t* ot_conv_3_4_3 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* ot_residual_12 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* wt_conv_3_5_1 = (scalar_t*)calloc(1*1*1024*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_5_1 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_5_1 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_5_2 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  batch_ft* bt_conv_3_5_2 = (batch_ft*)calloc(256, sizeof(batch_ft));
  scalar_t* ot_conv_3_5_2 = (scalar_t*)calloc(16*16*256, sizeof(scalar_t));
  scalar_t* wt_conv_3_5_3 = (scalar_t*)calloc(1*1*256*1024, sizeof(scalar_t));
  batch_ft* bt_conv_3_5_3 = (batch_ft*)calloc(1024, sizeof(batch_ft));
  scalar_t* ot_conv_3_5_3 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* ot_residual_13 = (scalar_t*)calloc(16*16*1024, sizeof(scalar_t));
  scalar_t* wt_conv_4_0_1 = (scalar_t*)calloc(1*1*1024*512, sizeof(scalar_t));
  batch_ft* bt_conv_4_0_1 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_4_0_1 = (scalar_t*)calloc(16*16*512, sizeof(scalar_t));
  scalar_t* wt_conv_4_0_2 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  batch_ft* bt_conv_4_0_2 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_4_0_2 = (scalar_t*)calloc(8*8*512, sizeof(scalar_t));
  scalar_t* wt_conv_4_0_3 = (scalar_t*)calloc(1*1*512*2048, sizeof(scalar_t));
  batch_ft* bt_conv_4_0_3 = (batch_ft*)calloc(2048, sizeof(batch_ft));
  scalar_t* ot_conv_4_0_3 = (scalar_t*)calloc(8*8*2048, sizeof(scalar_t));
  scalar_t* ot_residual_14 = (scalar_t*)calloc(8*8*2048, sizeof(scalar_t));
  scalar_t* wt_conv_4_1_1 = (scalar_t*)calloc(1*1*2048*512, sizeof(scalar_t));
  batch_ft* bt_conv_4_1_1 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_4_1_1 = (scalar_t*)calloc(8*8*512, sizeof(scalar_t));
  scalar_t* wt_conv_4_1_2 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  batch_ft* bt_conv_4_1_2 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_4_1_2 = (scalar_t*)calloc(8*8*512, sizeof(scalar_t));
  scalar_t* wt_conv_4_1_3 = (scalar_t*)calloc(1*1*512*2048, sizeof(scalar_t));
  batch_ft* bt_conv_4_1_3 = (batch_ft*)calloc(2048, sizeof(batch_ft));
  scalar_t* ot_conv_4_1_3 = (scalar_t*)calloc(8*8*2048, sizeof(scalar_t));
  scalar_t* ot_residual_15 = (scalar_t*)calloc(8*8*2048, sizeof(scalar_t));
  scalar_t* wt_conv_4_2_1 = (scalar_t*)calloc(1*1*2048*512, sizeof(scalar_t));
  batch_ft* bt_conv_4_2_1 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_4_2_1 = (scalar_t*)calloc(8*8*512, sizeof(scalar_t));
  scalar_t* wt_conv_4_2_2 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  batch_ft* bt_conv_4_2_2 = (batch_ft*)calloc(512, sizeof(batch_ft));
  scalar_t* ot_conv_4_2_2 = (scalar_t*)calloc(8*8*512, sizeof(scalar_t));
  scalar_t* wt_conv_4_2_3 = (scalar_t*)calloc(1*1*512*2048, sizeof(scalar_t));
  batch_ft* bt_conv_4_2_3 = (batch_ft*)calloc(2048, sizeof(batch_ft));
  scalar_t* ot_conv_4_2_3 = (scalar_t*)calloc(8*8*2048, sizeof(scalar_t));
  scalar_t* ot_residual_16 = (scalar_t*)calloc(8*8*2048, sizeof(scalar_t));
  scalar_t* ot_avgpol_1 = (scalar_t*)calloc(2048, sizeof(scalar_t));
  scalar_t* wt_connct_1 = (scalar_t*)calloc(2048*1000, sizeof(scalar_t));
  scalar_t* bs_connct_1 = (scalar_t*)calloc(1000, sizeof(scalar_t));
  scalar_t* ot_connct_1 = (scalar_t*)calloc(1*1*1000, sizeof(scalar_t));
  float* ot_softmax = (float*)calloc(1000, sizeof(float));

  // Read weights and bias and batch normalization factors from file
  fread((char*)wt_conv_1, sizeof(scalar_t), 7*7*3*64, wt_fp);
  fread((char*)bt_conv_1, sizeof(batch_ft), 64, wt_fp);
  fread((char*)wt_conv_1_0_1, sizeof(scalar_t), 1*1*64*64, wt_fp);
  fread((char*)bt_conv_1_0_1, sizeof(batch_ft), 64, wt_fp);
  fread((char*)wt_conv_1_0_2, sizeof(scalar_t), 3*3*64*64, wt_fp);
  fread((char*)bt_conv_1_0_2, sizeof(batch_ft), 64, wt_fp);
  fread((char*)wt_conv_1_0_3, sizeof(scalar_t), 1*1*64*256, wt_fp);
  fread((char*)bt_conv_1_0_3, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_1_1_1, sizeof(scalar_t), 1*1*256*64, wt_fp);
  fread((char*)bt_conv_1_1_1, sizeof(batch_ft), 64, wt_fp);
  fread((char*)wt_conv_1_1_2, sizeof(scalar_t), 3*3*64*64, wt_fp);
  fread((char*)bt_conv_1_1_2, sizeof(batch_ft), 64, wt_fp);
  fread((char*)wt_conv_1_1_3, sizeof(scalar_t), 1*1*64*256, wt_fp);
  fread((char*)bt_conv_1_1_3, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_1_2_1, sizeof(scalar_t), 1*1*256*64, wt_fp);
  fread((char*)bt_conv_1_2_1, sizeof(batch_ft), 64, wt_fp);
  fread((char*)wt_conv_1_2_2, sizeof(scalar_t), 3*3*64*64, wt_fp);
  fread((char*)bt_conv_1_2_2, sizeof(batch_ft), 64, wt_fp);
  fread((char*)wt_conv_1_2_3, sizeof(scalar_t), 1*1*64*256, wt_fp);
  fread((char*)bt_conv_1_2_3, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_2_0_1, sizeof(scalar_t), 1*1*256*128, wt_fp);
  fread((char*)bt_conv_2_0_1, sizeof(batch_ft), 128, wt_fp);
  fread((char*)wt_conv_2_0_2, sizeof(scalar_t), 3*3*128*128, wt_fp);
  fread((char*)bt_conv_2_0_2, sizeof(batch_ft), 128, wt_fp);
  fread((char*)wt_conv_2_0_3, sizeof(scalar_t), 1*1*128*512, wt_fp);
  fread((char*)bt_conv_2_0_3, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_2_1_1, sizeof(scalar_t), 1*1*512*128, wt_fp);
  fread((char*)bt_conv_2_1_1, sizeof(batch_ft), 128, wt_fp);
  fread((char*)wt_conv_2_1_2, sizeof(scalar_t), 3*3*128*128, wt_fp);
  fread((char*)bt_conv_2_1_2, sizeof(batch_ft), 128, wt_fp);
  fread((char*)wt_conv_2_1_3, sizeof(scalar_t), 1*1*128*512, wt_fp);
  fread((char*)bt_conv_2_1_3, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_2_2_1, sizeof(scalar_t), 1*1*512*128, wt_fp);
  fread((char*)bt_conv_2_2_1, sizeof(batch_ft), 128, wt_fp);
  fread((char*)wt_conv_2_2_2, sizeof(scalar_t), 3*3*128*128, wt_fp);
  fread((char*)bt_conv_2_2_2, sizeof(batch_ft), 128, wt_fp);
  fread((char*)wt_conv_2_2_3, sizeof(scalar_t), 1*1*128*512, wt_fp);
  fread((char*)bt_conv_2_2_3, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_2_3_1, sizeof(scalar_t), 1*1*512*128, wt_fp);
  fread((char*)bt_conv_2_3_1, sizeof(batch_ft), 128, wt_fp);
  fread((char*)wt_conv_2_3_2, sizeof(scalar_t), 3*3*128*128, wt_fp);
  fread((char*)bt_conv_2_3_2, sizeof(batch_ft), 128, wt_fp);
  fread((char*)wt_conv_2_3_3, sizeof(scalar_t), 1*1*128*512, wt_fp);
  fread((char*)bt_conv_2_3_3, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_3_0_1, sizeof(scalar_t), 1*1*512*256, wt_fp);
  fread((char*)bt_conv_3_0_1, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_0_2, sizeof(scalar_t), 3*3*256*256, wt_fp);
  fread((char*)bt_conv_3_0_2, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_0_3, sizeof(scalar_t), 1*1*256*1024, wt_fp);
  fread((char*)bt_conv_3_0_3, sizeof(batch_ft), 1024, wt_fp);
  fread((char*)wt_conv_3_1_1, sizeof(scalar_t), 1*1*1024*256, wt_fp);
  fread((char*)bt_conv_3_1_1, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_1_2, sizeof(scalar_t), 3*3*256*256, wt_fp);
  fread((char*)bt_conv_3_1_2, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_1_3, sizeof(scalar_t), 1*1*256*1024, wt_fp);
  fread((char*)bt_conv_3_1_3, sizeof(batch_ft), 1024, wt_fp);
  fread((char*)wt_conv_3_2_1, sizeof(scalar_t), 1*1*1024*256, wt_fp);
  fread((char*)bt_conv_3_2_1, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_2_2, sizeof(scalar_t), 3*3*256*256, wt_fp);
  fread((char*)bt_conv_3_2_2, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_2_3, sizeof(scalar_t), 1*1*256*1024, wt_fp);
  fread((char*)bt_conv_3_2_3, sizeof(batch_ft), 1024, wt_fp);
  fread((char*)wt_conv_3_3_1, sizeof(scalar_t), 1*1*1024*256, wt_fp);
  fread((char*)bt_conv_3_3_1, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_3_2, sizeof(scalar_t), 3*3*256*256, wt_fp);
  fread((char*)bt_conv_3_3_2, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_3_3, sizeof(scalar_t), 1*1*256*1024, wt_fp);
  fread((char*)bt_conv_3_3_3, sizeof(batch_ft), 1024, wt_fp);
  fread((char*)wt_conv_3_4_1, sizeof(scalar_t), 1*1*1024*256, wt_fp);
  fread((char*)bt_conv_3_4_1, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_4_2, sizeof(scalar_t), 3*3*256*256, wt_fp);
  fread((char*)bt_conv_3_4_2, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_4_3, sizeof(scalar_t), 1*1*256*1024, wt_fp);
  fread((char*)bt_conv_3_4_3, sizeof(batch_ft), 1024, wt_fp);
  fread((char*)wt_conv_3_5_1, sizeof(scalar_t), 1*1*1024*256, wt_fp);
  fread((char*)bt_conv_3_5_1, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_5_2, sizeof(scalar_t), 3*3*256*256, wt_fp);
  fread((char*)bt_conv_3_5_2, sizeof(batch_ft), 256, wt_fp);
  fread((char*)wt_conv_3_5_3, sizeof(scalar_t), 1*1*256*1024, wt_fp);
  fread((char*)bt_conv_3_5_3, sizeof(batch_ft), 1024, wt_fp);
  fread((char*)wt_conv_4_0_1, sizeof(scalar_t), 1*1*1024*512, wt_fp);
  fread((char*)bt_conv_4_0_1, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_4_0_2, sizeof(scalar_t), 3*3*512*512, wt_fp);
  fread((char*)bt_conv_4_0_2, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_4_0_3, sizeof(scalar_t), 1*1*512*2048, wt_fp);
  fread((char*)bt_conv_4_0_3, sizeof(batch_ft), 2048, wt_fp);
  fread((char*)wt_conv_4_1_1, sizeof(scalar_t), 1*1*2048*512, wt_fp);
  fread((char*)bt_conv_4_1_1, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_4_1_2, sizeof(scalar_t), 3*3*512*512, wt_fp);
  fread((char*)bt_conv_4_1_2, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_4_1_3, sizeof(scalar_t), 1*1*512*2048, wt_fp);
  fread((char*)bt_conv_4_1_3, sizeof(batch_ft), 2048, wt_fp);
  fread((char*)wt_conv_4_2_1, sizeof(scalar_t), 1*1*2048*512, wt_fp);
  fread((char*)bt_conv_4_2_1, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_4_2_2, sizeof(scalar_t), 3*3*512*512, wt_fp);
  fread((char*)bt_conv_4_2_2, sizeof(batch_ft), 512, wt_fp);
  fread((char*)wt_conv_4_2_3, sizeof(scalar_t), 1*1*512*2048, wt_fp);
  fread((char*)bt_conv_4_2_3, sizeof(batch_ft), 2048, wt_fp);
  fread((char*)wt_connct_1, sizeof(scalar_t), 2048*1000, wt_fp);
  fread((char*)bs_connct_1, sizeof(scalar_t), 1000, wt_fp);

  fclose(wt_fp);

#ifdef PLANNER
  scalar_t* in_mem_0 = (scalar_t*)calloc(IN_MEM_SIZE, sizeof(scalar_t));
  scalar_t* wt_mem_0 = (scalar_t*)calloc(WT_MEM_SIZE, sizeof(scalar_t));
  scalar_t* ot_mem_0 = (scalar_t*)calloc(OT_MEM_SIZE, sizeof(scalar_t));
#endif

  clock_t start, end;

  // Run Network
  start = clock();

#ifdef PLANNER
  conv_1(im.data, wt_conv_1, ot_conv_1, 64, 3, 7, 256, 256, 2, 3, 3, 3, 3, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(im.data, wt_conv_1, ot_conv_1, 64, 3, 7, 256, 256, 2, 3, 3, 3, 3);
#endif
  batchnorm(ot_conv_1, ot_conv_1, bt_conv_1, 1, 64, 128, 128);
  leaky_relu(ot_conv_1, ot_conv_1, 1*64*128*128);

  maxpool2d(ot_conv_1, ot_maxpol_1, 64, 128, 128, 2, 2, 1);

#ifdef PLANNER
  conv_1_0_1(ot_maxpol_1, wt_conv_1_0_1, ot_conv_1_0_1, 64, 64, 1, 64, 64, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_maxpol_1, wt_conv_1_0_1, ot_conv_1_0_1, 64, 64, 1, 64, 64, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_1_0_1, ot_conv_1_0_1, bt_conv_1_0_1, 1, 64, 64, 64);
  leaky_relu(ot_conv_1_0_1, ot_conv_1_0_1, 1*64*64*64);

#ifdef PLANNER
  conv_1_0_2(ot_conv_1_0_1, wt_conv_1_0_2, ot_conv_1_0_2, 64, 64, 3, 64, 64, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_1_0_1, wt_conv_1_0_2, ot_conv_1_0_2, 64, 64, 3, 64, 64, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_1_0_2, ot_conv_1_0_2, bt_conv_1_0_2, 1, 64, 64, 64);
  leaky_relu(ot_conv_1_0_2, ot_conv_1_0_2, 1*64*64*64);

#ifdef PLANNER
  conv_1_0_3(ot_conv_1_0_2, wt_conv_1_0_3, ot_conv_1_0_3, 256, 64, 1, 64, 64, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_1_0_2, wt_conv_1_0_3, ot_conv_1_0_3, 256, 64, 1, 64, 64, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_1_0_3, ot_conv_1_0_3, bt_conv_1_0_3, 1, 256, 64, 64);
  residual(ot_maxpol_1, ot_conv_1_0_3, ot_residual_1, 1, 64, 64, 64, 256, 64, 64);
  leaky_relu(ot_residual_1, ot_residual_1, 1*256*64*64);

#ifdef PLANNER
  conv_1_1_1(ot_residual_1, wt_conv_1_1_1, ot_conv_1_1_1, 64, 256, 1, 64, 64, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_1, wt_conv_1_1_1, ot_conv_1_1_1, 64, 256, 1, 64, 64, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_1_1_1, ot_conv_1_1_1, bt_conv_1_1_1, 1, 64, 64, 64);
  leaky_relu(ot_conv_1_1_1, ot_conv_1_1_1, 1*64*64*64);

#ifdef PLANNER
  conv_1_1_2(ot_conv_1_1_1, wt_conv_1_1_2, ot_conv_1_1_2, 64, 64, 3, 64, 64, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_1_1_1, wt_conv_1_1_2, ot_conv_1_1_2, 64, 64, 3, 64, 64, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_1_1_2, ot_conv_1_1_2, bt_conv_1_1_2, 1, 64, 64, 64);
  leaky_relu(ot_conv_1_1_2, ot_conv_1_1_2, 1*64*64*64);

#ifdef PLANNER
  conv_1_1_3(ot_conv_1_1_2, wt_conv_1_1_3, ot_conv_1_1_3, 256, 64, 1, 64, 64, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_1_1_2, wt_conv_1_1_3, ot_conv_1_1_3, 256, 64, 1, 64, 64, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_1_1_3, ot_conv_1_1_3, bt_conv_1_1_3, 1, 256, 64, 64);
  residual(ot_residual_1, ot_conv_1_1_3, ot_residual_2, 1, 256, 64, 64, 256, 64, 64);
  leaky_relu(ot_residual_2, ot_residual_2, 1*256*64*64);

#ifdef PLANNER
  conv_1_2_1(ot_residual_2, wt_conv_1_2_1, ot_conv_1_2_1, 64, 256, 1, 64, 64, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_2, wt_conv_1_2_1, ot_conv_1_2_1, 64, 256, 1, 64, 64, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_1_2_1, ot_conv_1_2_1, bt_conv_1_2_1, 1, 64, 64, 64);
  leaky_relu(ot_conv_1_2_1, ot_conv_1_2_1, 1*64*64*64);

#ifdef PLANNER
  conv_1_2_2(ot_conv_1_2_1, wt_conv_1_2_2, ot_conv_1_2_2, 64, 64, 3, 64, 64, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_1_2_1, wt_conv_1_2_2, ot_conv_1_2_2, 64, 64, 3, 64, 64, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_1_2_2, ot_conv_1_2_2, bt_conv_1_2_2, 1, 64, 64, 64);
  leaky_relu(ot_conv_1_2_2, ot_conv_1_2_2, 1*64*64*64);

#ifdef PLANNER
  conv_1_2_3(ot_conv_1_2_2, wt_conv_1_2_3, ot_conv_1_2_3, 256, 64, 1, 64, 64, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_1_2_2, wt_conv_1_2_3, ot_conv_1_2_3, 256, 64, 1, 64, 64, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_1_2_3, ot_conv_1_2_3, bt_conv_1_2_3, 1, 256, 64, 64);
  residual(ot_residual_2, ot_conv_1_2_3, ot_residual_3, 1, 256, 64, 64, 256, 64, 64);
  leaky_relu(ot_residual_3, ot_residual_3, 1*256*64*64);

#ifdef PLANNER
  conv_2_0_1(ot_residual_3, wt_conv_2_0_1, ot_conv_2_0_1, 128, 256, 1, 64, 64, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_3, wt_conv_2_0_1, ot_conv_2_0_1, 128, 256, 1, 64, 64, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_2_0_1, ot_conv_2_0_1, bt_conv_2_0_1, 1, 128, 64, 64);
  leaky_relu(ot_conv_2_0_1, ot_conv_2_0_1, 1*128*64*64);

#ifdef PLANNER
  conv_2_0_2(ot_conv_2_0_1, wt_conv_2_0_2, ot_conv_2_0_2, 128, 128, 3, 64, 64, 2, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_2_0_1, wt_conv_2_0_2, ot_conv_2_0_2, 128, 128, 3, 64, 64, 2, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_2_0_2, ot_conv_2_0_2, bt_conv_2_0_2, 1, 128, 32, 32);
  leaky_relu(ot_conv_2_0_2, ot_conv_2_0_2, 1*128*32*32);

#ifdef PLANNER
  conv_2_0_3(ot_conv_2_0_2, wt_conv_2_0_3, ot_conv_2_0_3, 512, 128, 1, 32, 32, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_2_0_2, wt_conv_2_0_3, ot_conv_2_0_3, 512, 128, 1, 32, 32, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_2_0_3, ot_conv_2_0_3, bt_conv_2_0_3, 1, 512, 32, 32);
  residual(ot_residual_3, ot_conv_2_0_3, ot_residual_4, 1, 256, 64, 64, 512, 32, 32);
  leaky_relu(ot_residual_4, ot_residual_4, 1*512*32*32);

#ifdef PLANNER
  conv_2_1_1(ot_residual_4, wt_conv_2_1_1, ot_conv_2_1_1, 128, 512, 1, 32, 32, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_4, wt_conv_2_1_1, ot_conv_2_1_1, 128, 512, 1, 32, 32, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_2_1_1, ot_conv_2_1_1, bt_conv_2_1_1, 1, 128, 32, 32);
  leaky_relu(ot_conv_2_1_1, ot_conv_2_1_1, 1*128*32*32);

#ifdef PLANNER
  conv_2_1_2(ot_conv_2_1_1, wt_conv_2_1_2, ot_conv_2_1_2, 128, 128, 3, 32, 32, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_2_1_1, wt_conv_2_1_2, ot_conv_2_1_2, 128, 128, 3, 32, 32, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_2_1_2, ot_conv_2_1_2, bt_conv_2_1_2, 1, 128, 32, 32);
  leaky_relu(ot_conv_2_1_2, ot_conv_2_1_2, 1*128*32*32);

#ifdef PLANNER
  conv_2_1_3(ot_conv_2_1_2, wt_conv_2_1_3, ot_conv_2_1_3, 512, 128, 1, 32, 32, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_2_1_2, wt_conv_2_1_3, ot_conv_2_1_3, 512, 128, 1, 32, 32, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_2_1_3, ot_conv_2_1_3, bt_conv_2_1_3, 1, 512, 32, 32);
  residual(ot_residual_4, ot_conv_2_1_3, ot_residual_5, 1, 512, 32, 32, 512, 32, 32);
  leaky_relu(ot_residual_5, ot_residual_5, 1*512*32*32);

#ifdef PLANNER
  conv_2_2_1(ot_residual_5, wt_conv_2_2_1, ot_conv_2_2_1, 128, 512, 1, 32, 32, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_5, wt_conv_2_2_1, ot_conv_2_2_1, 128, 512, 1, 32, 32, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_2_2_1, ot_conv_2_2_1, bt_conv_2_2_1, 1, 128, 32, 32);
  leaky_relu(ot_conv_2_2_1, ot_conv_2_2_1, 1*128*32*32);

#ifdef PLANNER
  conv_2_2_2(ot_conv_2_2_1, wt_conv_2_2_2, ot_conv_2_2_2, 128, 128, 3, 32, 32, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_2_2_1, wt_conv_2_2_2, ot_conv_2_2_2, 128, 128, 3, 32, 32, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_2_2_2, ot_conv_2_2_2, bt_conv_2_2_2, 1, 128, 32, 32);
  leaky_relu(ot_conv_2_2_2, ot_conv_2_2_2, 1*128*32*32);

#ifdef PLANNER
  conv_2_2_3(ot_conv_2_2_2, wt_conv_2_2_3, ot_conv_2_2_3, 512, 128, 1, 32, 32, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_2_2_2, wt_conv_2_2_3, ot_conv_2_2_3, 512, 128, 1, 32, 32, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_2_2_3, ot_conv_2_2_3, bt_conv_2_2_3, 1, 512, 32, 32);
  residual(ot_residual_5, ot_conv_2_2_3, ot_residual_6, 1, 512, 32, 32, 512, 32, 32);
  leaky_relu(ot_residual_6, ot_residual_6, 1*512*32*32);

#ifdef PLANNER
  conv_2_3_1(ot_residual_6, wt_conv_2_3_1, ot_conv_2_3_1, 128, 512, 1, 32, 32, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_6, wt_conv_2_3_1, ot_conv_2_3_1, 128, 512, 1, 32, 32, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_2_3_1, ot_conv_2_3_1, bt_conv_2_3_1, 1, 128, 32, 32);
  leaky_relu(ot_conv_2_3_1, ot_conv_2_3_1, 1*128*32*32);

#ifdef PLANNER
  conv_2_3_2(ot_conv_2_3_1, wt_conv_2_3_2, ot_conv_2_3_2, 128, 128, 3, 32, 32, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_2_3_1, wt_conv_2_3_2, ot_conv_2_3_2, 128, 128, 3, 32, 32, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_2_3_2, ot_conv_2_3_2, bt_conv_2_3_2, 1, 128, 32, 32);
  leaky_relu(ot_conv_2_3_2, ot_conv_2_3_2, 1*128*32*32);

#ifdef PLANNER
  conv_2_3_3(ot_conv_2_3_2, wt_conv_2_3_3, ot_conv_2_3_3, 512, 128, 1, 32, 32, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_2_3_2, wt_conv_2_3_3, ot_conv_2_3_3, 512, 128, 1, 32, 32, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_2_3_3, ot_conv_2_3_3, bt_conv_2_3_3, 1, 512, 32, 32);
  residual(ot_residual_6, ot_conv_2_3_3, ot_residual_7, 1, 512, 32, 32, 512, 32, 32);
  leaky_relu(ot_residual_7, ot_residual_7, 1*512*32*32);

#ifdef PLANNER
  conv_3_0_1(ot_residual_7, wt_conv_3_0_1, ot_conv_3_0_1, 256, 512, 1, 32, 32, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_7, wt_conv_3_0_1, ot_conv_3_0_1, 256, 512, 1, 32, 32, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_0_1, ot_conv_3_0_1, bt_conv_3_0_1, 1, 256, 32, 32);
  leaky_relu(ot_conv_3_0_1, ot_conv_3_0_1, 1*256*32*32);

#ifdef PLANNER
  conv_3_0_2(ot_conv_3_0_1, wt_conv_3_0_2, ot_conv_3_0_2, 256, 256, 3, 32, 32, 2, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_0_1, wt_conv_3_0_2, ot_conv_3_0_2, 256, 256, 3, 32, 32, 2, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_3_0_2, ot_conv_3_0_2, bt_conv_3_0_2, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_0_2, ot_conv_3_0_2, 1*256*16*16);

#ifdef PLANNER
  conv_3_0_3(ot_conv_3_0_2, wt_conv_3_0_3, ot_conv_3_0_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_0_2, wt_conv_3_0_3, ot_conv_3_0_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_0_3, ot_conv_3_0_3, bt_conv_3_0_3, 1, 1024, 16, 16);
  residual(ot_residual_7, ot_conv_3_0_3, ot_residual_8, 1, 512, 32, 32, 1024, 16, 16);
  leaky_relu(ot_residual_8, ot_residual_8, 1*1024*16*16);

#ifdef PLANNER
  conv_3_1_1(ot_residual_8, wt_conv_3_1_1, ot_conv_3_1_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_8, wt_conv_3_1_1, ot_conv_3_1_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_1_1, ot_conv_3_1_1, bt_conv_3_1_1, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_1_1, ot_conv_3_1_1, 1*256*16*16);

#ifdef PLANNER
  conv_3_1_2(ot_conv_3_1_1, wt_conv_3_1_2, ot_conv_3_1_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_1_1, wt_conv_3_1_2, ot_conv_3_1_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_3_1_2, ot_conv_3_1_2, bt_conv_3_1_2, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_1_2, ot_conv_3_1_2, 1*256*16*16);

#ifdef PLANNER
  conv_3_1_3(ot_conv_3_1_2, wt_conv_3_1_3, ot_conv_3_1_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_1_2, wt_conv_3_1_3, ot_conv_3_1_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_1_3, ot_conv_3_1_3, bt_conv_3_1_3, 1, 1024, 16, 16);
  residual(ot_residual_8, ot_conv_3_1_3, ot_residual_9, 1, 1024, 16, 16, 1024, 16, 16);
  leaky_relu(ot_residual_9, ot_residual_9, 1*1024*16*16);

#ifdef PLANNER
  conv_3_2_1(ot_residual_9, wt_conv_3_2_1, ot_conv_3_2_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_9, wt_conv_3_2_1, ot_conv_3_2_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_2_1, ot_conv_3_2_1, bt_conv_3_2_1, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_2_1, ot_conv_3_2_1, 1*256*16*16);

#ifdef PLANNER
  conv_3_2_2(ot_conv_3_2_1, wt_conv_3_2_2, ot_conv_3_2_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_2_1, wt_conv_3_2_2, ot_conv_3_2_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_3_2_2, ot_conv_3_2_2, bt_conv_3_2_2, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_2_2, ot_conv_3_2_2, 1*256*16*16);

#ifdef PLANNER
  conv_3_2_3(ot_conv_3_2_2, wt_conv_3_2_3, ot_conv_3_2_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_2_2, wt_conv_3_2_3, ot_conv_3_2_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_2_3, ot_conv_3_2_3, bt_conv_3_2_3, 1, 1024, 16, 16);
  residual(ot_residual_9, ot_conv_3_2_3, ot_residual_10, 1, 1024, 16, 16, 1024, 16, 16);
  leaky_relu(ot_residual_10, ot_residual_10, 1*1024*16*16);

#ifdef PLANNER
  conv_3_3_1(ot_residual_10, wt_conv_3_3_1, ot_conv_3_3_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_10, wt_conv_3_3_1, ot_conv_3_3_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_3_1, ot_conv_3_3_1, bt_conv_3_3_1, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_3_1, ot_conv_3_3_1, 1*256*16*16);

#ifdef PLANNER
  conv_3_3_2(ot_conv_3_3_1, wt_conv_3_3_2, ot_conv_3_3_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_3_1, wt_conv_3_3_2, ot_conv_3_3_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_3_3_2, ot_conv_3_3_2, bt_conv_3_3_2, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_3_2, ot_conv_3_3_2, 1*256*16*16);

#ifdef PLANNER
  conv_3_3_3(ot_conv_3_3_2, wt_conv_3_3_3, ot_conv_3_3_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_3_2, wt_conv_3_3_3, ot_conv_3_3_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_3_3, ot_conv_3_3_3, bt_conv_3_3_3, 1, 1024, 16, 16);
  residual(ot_residual_10, ot_conv_3_3_3, ot_residual_11, 1, 1024, 16, 16, 1024, 16, 16);
  leaky_relu(ot_residual_11, ot_residual_11, 1*1024*16*16);

#ifdef PLANNER
  conv_3_4_1(ot_residual_11, wt_conv_3_4_1, ot_conv_3_4_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_11, wt_conv_3_4_1, ot_conv_3_4_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_4_1, ot_conv_3_4_1, bt_conv_3_4_1, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_4_1, ot_conv_3_4_1, 1*256*16*16);

#ifdef PLANNER
  conv_3_4_2(ot_conv_3_4_1, wt_conv_3_4_2, ot_conv_3_4_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_4_1, wt_conv_3_4_2, ot_conv_3_4_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_3_4_2, ot_conv_3_4_2, bt_conv_3_4_2, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_4_2, ot_conv_3_4_2, 1*256*16*16);

#ifdef PLANNER
  conv_3_4_3(ot_conv_3_4_2, wt_conv_3_4_3, ot_conv_3_4_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_4_2, wt_conv_3_4_3, ot_conv_3_4_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_4_3, ot_conv_3_4_3, bt_conv_3_4_3, 1, 1024, 16, 16);
  residual(ot_residual_11, ot_conv_3_4_3, ot_residual_12, 1, 1024, 16, 16, 1024, 16, 16);
  leaky_relu(ot_residual_12, ot_residual_12, 1*1024*16*16);

#ifdef PLANNER
  conv_3_5_1(ot_residual_12, wt_conv_3_5_1, ot_conv_3_5_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_12, wt_conv_3_5_1, ot_conv_3_5_1, 256, 1024, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_5_1, ot_conv_3_5_1, bt_conv_3_5_1, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_5_1, ot_conv_3_5_1, 1*256*16*16);

#ifdef PLANNER
  conv_3_5_2(ot_conv_3_5_1, wt_conv_3_5_2, ot_conv_3_5_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_5_1, wt_conv_3_5_2, ot_conv_3_5_2, 256, 256, 3, 16, 16, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_3_5_2, ot_conv_3_5_2, bt_conv_3_5_2, 1, 256, 16, 16);
  leaky_relu(ot_conv_3_5_2, ot_conv_3_5_2, 1*256*16*16);

#ifdef PLANNER
  conv_3_5_3(ot_conv_3_5_2, wt_conv_3_5_3, ot_conv_3_5_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_3_5_2, wt_conv_3_5_3, ot_conv_3_5_3, 1024, 256, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_3_5_3, ot_conv_3_5_3, bt_conv_3_5_3, 1, 1024, 16, 16);
  residual(ot_residual_12, ot_conv_3_5_3, ot_residual_13, 1, 1024, 16, 16, 1024, 16, 16);
  leaky_relu(ot_residual_13, ot_residual_13, 1*1024*16*16);

#ifdef PLANNER
  conv_4_0_1(ot_residual_13, wt_conv_4_0_1, ot_conv_4_0_1, 512, 1024, 1, 16, 16, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_13, wt_conv_4_0_1, ot_conv_4_0_1, 512, 1024, 1, 16, 16, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_4_0_1, ot_conv_4_0_1, bt_conv_4_0_1, 1, 512, 16, 16);
  leaky_relu(ot_conv_4_0_1, ot_conv_4_0_1, 1*512*16*16);

#ifdef PLANNER
  conv_4_0_2(ot_conv_4_0_1, wt_conv_4_0_2, ot_conv_4_0_2, 512, 512, 3, 16, 16, 2, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_4_0_1, wt_conv_4_0_2, ot_conv_4_0_2, 512, 512, 3, 16, 16, 2, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_4_0_2, ot_conv_4_0_2, bt_conv_4_0_2, 1, 512, 8, 8);
  leaky_relu(ot_conv_4_0_2, ot_conv_4_0_2, 1*512*8*8);

#ifdef PLANNER
  conv_4_0_3(ot_conv_4_0_2, wt_conv_4_0_3, ot_conv_4_0_3, 2048, 512, 1, 8, 8, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_4_0_2, wt_conv_4_0_3, ot_conv_4_0_3, 2048, 512, 1, 8, 8, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_4_0_3, ot_conv_4_0_3, bt_conv_4_0_3, 1, 2048, 8, 8);
  residual(ot_residual_13, ot_conv_4_0_3, ot_residual_14, 1, 1024, 16, 16, 2048, 8, 8);
  leaky_relu(ot_residual_14, ot_residual_14, 1*2048*8*8);

#ifdef PLANNER
  conv_4_1_1(ot_residual_14, wt_conv_4_1_1, ot_conv_4_1_1, 512, 2048, 1, 8, 8, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_14, wt_conv_4_1_1, ot_conv_4_1_1, 512, 2048, 1, 8, 8, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_4_1_1, ot_conv_4_1_1, bt_conv_4_1_1, 1, 512, 8, 8);
  leaky_relu(ot_conv_4_1_1, ot_conv_4_1_1, 1*512*8*8);

#ifdef PLANNER
  conv_4_1_2(ot_conv_4_1_1, wt_conv_4_1_2, ot_conv_4_1_2, 512, 512, 3, 8, 8, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_4_1_1, wt_conv_4_1_2, ot_conv_4_1_2, 512, 512, 3, 8, 8, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_4_1_2, ot_conv_4_1_2, bt_conv_4_1_2, 1, 512, 8, 8);
  leaky_relu(ot_conv_4_1_2, ot_conv_4_1_2, 1*512*8*8);

#ifdef PLANNER
  conv_4_1_3(ot_conv_4_1_2, wt_conv_4_1_3, ot_conv_4_1_3, 2048, 512, 1, 8, 8, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_4_1_2, wt_conv_4_1_3, ot_conv_4_1_3, 2048, 512, 1, 8, 8, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_4_1_3, ot_conv_4_1_3, bt_conv_4_1_3, 1, 2048, 8, 8);
  residual(ot_residual_14, ot_conv_4_1_3, ot_residual_15, 1, 2048, 8, 8, 2048, 8, 8);
  leaky_relu(ot_residual_15, ot_residual_15, 1*2048*8*8);

#ifdef PLANNER
  conv_4_2_1(ot_residual_15, wt_conv_4_2_1, ot_conv_4_2_1, 512, 2048, 1, 8, 8, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_residual_15, wt_conv_4_2_1, ot_conv_4_2_1, 512, 2048, 1, 8, 8, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_4_2_1, ot_conv_4_2_1, bt_conv_4_2_1, 1, 512, 8, 8);
  leaky_relu(ot_conv_4_2_1, ot_conv_4_2_1, 1*512*8*8);

#ifdef PLANNER
  conv_4_2_2(ot_conv_4_2_1, wt_conv_4_2_2, ot_conv_4_2_2, 512, 512, 3, 8, 8, 1, 1, 1, 1, 1, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_4_2_1, wt_conv_4_2_2, ot_conv_4_2_2, 512, 512, 3, 8, 8, 1, 1, 1, 1, 1);
#endif
  batchnorm(ot_conv_4_2_2, ot_conv_4_2_2, bt_conv_4_2_2, 1, 512, 8, 8);
  leaky_relu(ot_conv_4_2_2, ot_conv_4_2_2, 1*512*8*8);

#ifdef PLANNER
  conv_4_2_3(ot_conv_4_2_2, wt_conv_4_2_3, ot_conv_4_2_3, 2048, 512, 1, 8, 8, 1, 0, 0, 0, 0, in_mem_0, wt_mem_0, ot_mem_0);
#else
  conv2d(ot_conv_4_2_2, wt_conv_4_2_3, ot_conv_4_2_3, 2048, 512, 1, 8, 8, 1, 0, 0, 0, 0);
#endif
  batchnorm(ot_conv_4_2_3, ot_conv_4_2_3, bt_conv_4_2_3, 1, 2048, 8, 8);
  residual(ot_residual_15, ot_conv_4_2_3, ot_residual_16, 1, 2048, 8, 8, 2048, 8, 8);
  leaky_relu(ot_residual_16, ot_residual_16, 1*2048*8*8);

  avgpool2d(ot_residual_16, ot_avgpol_1, 2048, 8, 8);

  connected(ot_avgpol_1, wt_connct_1, ot_connct_1, 1000, 2048);
  addbias_connected(ot_connct_1, ot_connct_1, bs_connct_1, 1000);
  softmax(ot_connct_1, ot_softmax, 1000, 1.000000);

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
  free(wt_conv_1);
  free(bt_conv_1);
  free(ot_conv_1);
  free(ot_maxpol_1);
  free(wt_conv_1_0_1);
  free(bt_conv_1_0_1);
  free(ot_conv_1_0_1);
  free(wt_conv_1_0_2);
  free(bt_conv_1_0_2);
  free(ot_conv_1_0_2);
  free(wt_conv_1_0_3);
  free(bt_conv_1_0_3);
  free(ot_conv_1_0_3);
  free(ot_residual_1);
  free(wt_conv_1_1_1);
  free(bt_conv_1_1_1);
  free(ot_conv_1_1_1);
  free(wt_conv_1_1_2);
  free(bt_conv_1_1_2);
  free(ot_conv_1_1_2);
  free(wt_conv_1_1_3);
  free(bt_conv_1_1_3);
  free(ot_conv_1_1_3);
  free(ot_residual_2);
  free(wt_conv_1_2_1);
  free(bt_conv_1_2_1);
  free(ot_conv_1_2_1);
  free(wt_conv_1_2_2);
  free(bt_conv_1_2_2);
  free(ot_conv_1_2_2);
  free(wt_conv_1_2_3);
  free(bt_conv_1_2_3);
  free(ot_conv_1_2_3);
  free(ot_residual_3);
  free(wt_conv_2_0_1);
  free(bt_conv_2_0_1);
  free(ot_conv_2_0_1);
  free(wt_conv_2_0_2);
  free(bt_conv_2_0_2);
  free(ot_conv_2_0_2);
  free(wt_conv_2_0_3);
  free(bt_conv_2_0_3);
  free(ot_conv_2_0_3);
  free(ot_residual_4);
  free(wt_conv_2_1_1);
  free(bt_conv_2_1_1);
  free(ot_conv_2_1_1);
  free(wt_conv_2_1_2);
  free(bt_conv_2_1_2);
  free(ot_conv_2_1_2);
  free(wt_conv_2_1_3);
  free(bt_conv_2_1_3);
  free(ot_conv_2_1_3);
  free(ot_residual_5);
  free(wt_conv_2_2_1);
  free(bt_conv_2_2_1);
  free(ot_conv_2_2_1);
  free(wt_conv_2_2_2);
  free(bt_conv_2_2_2);
  free(ot_conv_2_2_2);
  free(wt_conv_2_2_3);
  free(bt_conv_2_2_3);
  free(ot_conv_2_2_3);
  free(ot_residual_6);
  free(wt_conv_2_3_1);
  free(bt_conv_2_3_1);
  free(ot_conv_2_3_1);
  free(wt_conv_2_3_2);
  free(bt_conv_2_3_2);
  free(ot_conv_2_3_2);
  free(wt_conv_2_3_3);
  free(bt_conv_2_3_3);
  free(ot_conv_2_3_3);
  free(ot_residual_7);
  free(wt_conv_3_0_1);
  free(bt_conv_3_0_1);
  free(ot_conv_3_0_1);
  free(wt_conv_3_0_2);
  free(bt_conv_3_0_2);
  free(ot_conv_3_0_2);
  free(wt_conv_3_0_3);
  free(bt_conv_3_0_3);
  free(ot_conv_3_0_3);
  free(ot_residual_8);
  free(wt_conv_3_1_1);
  free(bt_conv_3_1_1);
  free(ot_conv_3_1_1);
  free(wt_conv_3_1_2);
  free(bt_conv_3_1_2);
  free(ot_conv_3_1_2);
  free(wt_conv_3_1_3);
  free(bt_conv_3_1_3);
  free(ot_conv_3_1_3);
  free(ot_residual_9);
  free(wt_conv_3_2_1);
  free(bt_conv_3_2_1);
  free(ot_conv_3_2_1);
  free(wt_conv_3_2_2);
  free(bt_conv_3_2_2);
  free(ot_conv_3_2_2);
  free(wt_conv_3_2_3);
  free(bt_conv_3_2_3);
  free(ot_conv_3_2_3);
  free(ot_residual_10);
  free(wt_conv_3_3_1);
  free(bt_conv_3_3_1);
  free(ot_conv_3_3_1);
  free(wt_conv_3_3_2);
  free(bt_conv_3_3_2);
  free(ot_conv_3_3_2);
  free(wt_conv_3_3_3);
  free(bt_conv_3_3_3);
  free(ot_conv_3_3_3);
  free(ot_residual_11);
  free(wt_conv_3_4_1);
  free(bt_conv_3_4_1);
  free(ot_conv_3_4_1);
  free(wt_conv_3_4_2);
  free(bt_conv_3_4_2);
  free(ot_conv_3_4_2);
  free(wt_conv_3_4_3);
  free(bt_conv_3_4_3);
  free(ot_conv_3_4_3);
  free(ot_residual_12);
  free(wt_conv_3_5_1);
  free(bt_conv_3_5_1);
  free(ot_conv_3_5_1);
  free(wt_conv_3_5_2);
  free(bt_conv_3_5_2);
  free(ot_conv_3_5_2);
  free(wt_conv_3_5_3);
  free(bt_conv_3_5_3);
  free(ot_conv_3_5_3);
  free(ot_residual_13);
  free(wt_conv_4_0_1);
  free(bt_conv_4_0_1);
  free(ot_conv_4_0_1);
  free(wt_conv_4_0_2);
  free(bt_conv_4_0_2);
  free(ot_conv_4_0_2);
  free(wt_conv_4_0_3);
  free(bt_conv_4_0_3);
  free(ot_conv_4_0_3);
  free(ot_residual_14);
  free(wt_conv_4_1_1);
  free(bt_conv_4_1_1);
  free(ot_conv_4_1_1);
  free(wt_conv_4_1_2);
  free(bt_conv_4_1_2);
  free(ot_conv_4_1_2);
  free(wt_conv_4_1_3);
  free(bt_conv_4_1_3);
  free(ot_conv_4_1_3);
  free(ot_residual_15);
  free(wt_conv_4_2_1);
  free(bt_conv_4_2_1);
  free(ot_conv_4_2_1);
  free(wt_conv_4_2_2);
  free(bt_conv_4_2_2);
  free(ot_conv_4_2_2);
  free(wt_conv_4_2_3);
  free(bt_conv_4_2_3);
  free(ot_conv_4_2_3);
  free(ot_residual_16);
  free(ot_avgpol_1);
  free(wt_connct_1);
  free(bs_connct_1);
  free(ot_connct_1);
  free(ot_softmax);

  return 0;
}
