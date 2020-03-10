#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "ops.h"
#include "type.h"
#include "image.h"
#include "classifier.h"
#include "topk.h"
#include "crop.h"
#include "client.h"

#define TOPK  5

int main(int argc, char* argv[])
{
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <image file path>.jpg <top k> <server ip> <server port>\n", argv[0]);
    exit(0);
  }

  int top_K = atoi(argv[2]);

  // Image load
  image raw_img = load_image(argv[1], 256, 256, 3);
  //store_image("boxed_dog.jpg", im);
  image im = crop(raw_img, 224, 224);

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
  scalar_t* wt_connct_1 = (scalar_t*)calloc(7*7*512*4096, sizeof(scalar_t));
  scalar_t* bs_connct_1 = (scalar_t*)calloc(4096, sizeof(scalar_t));
  scalar_t* ot_connct_1 = (scalar_t*)calloc(4096, sizeof(scalar_t));
  scalar_t* wt_connct_2 = (scalar_t*)calloc(4096*4096, sizeof(scalar_t));
  scalar_t* bs_connct_2 = (scalar_t*)calloc(4096, sizeof(scalar_t));
  scalar_t* ot_connct_2 = (scalar_t*)calloc(4096, sizeof(scalar_t));
  scalar_t* wt_connct_3 = (scalar_t*)calloc(4096*1000, sizeof(scalar_t));
  scalar_t* bs_connct_3 = (scalar_t*)calloc(1000, sizeof(scalar_t));
  scalar_t* ot_connct_3 = (scalar_t*)calloc(1000, sizeof(scalar_t));
  scalar_t* ot_softmax  = (scalar_t*)calloc(1000, sizeof(float));

  int server_fd;
  struct sockaddr_in server_addr;

  if (init_client(&server_addr, &server_fd, argv[3], atoi(argv[4])) == CLIENT_FAIL) {
    fprintf(stderr, "Client : Cannot create socket.\n");
    exit(0);
  }

  if (connect(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr))) {
    fprintf(stderr, "Client : Cannot connect to server.\n");
    exit(0);
  }

  size_t rd_size;
  clock_t start, end;
  // Run Network
  start = clock();

  rd_size = recv(server_fd, wt_conv_1_1, sizeof(scalar_t)*64*3*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_1_1: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_1_1, sizeof(scalar_t)*64, MSG_WAITALL);
  fprintf(stderr, "bs_conv_1_1: %ld\n", rd_size);

  conv2d(im.data, wt_conv_1_1, ot_conv_1_1, 64, 3, 3, 224, 224, 1, 1);
  addbias(ot_conv_1_1, ot_conv_1_1, bs_conv_1_1, 64, 224*224);
  relu(ot_conv_1_1, ot_conv_1_1, 64*224*224);

  fprintf(stderr, "conv_1_1 output: %f\n", ot_conv_1_1[0]);

  rd_size = recv(server_fd, wt_conv_1_2, sizeof(scalar_t)*64*64*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_1_2: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_1_2, sizeof(scalar_t)*64, MSG_WAITALL);
  fprintf(stderr, "bs_conv_1_2: %ld\n", rd_size);

  fprintf(stderr, "conv_1_2 weight: %f\n", wt_conv_1_2[0]);

  conv2d(ot_conv_1_1, wt_conv_1_2, ot_conv_1_2, 64, 64, 3, 224, 224, 1, 1);
  addbias(ot_conv_1_2, ot_conv_1_2, bs_conv_1_2, 64, 224*224);
  relu(ot_conv_1_2, ot_conv_1_2, 64*224*224);

  fprintf(stderr, "conv_1_2 output: %f\n", ot_conv_1_2[0]);

  maxpool2d(ot_conv_1_2, ot_maxpol_1, 64, 224, 224, 2, 2, 0);

  fprintf(stderr, "maxpool_1 output: %f\n", ot_maxpol_1[0]);

  rd_size = recv(server_fd, wt_conv_2_1, sizeof(scalar_t)*128*64*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_2_1: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_2_1, sizeof(scalar_t)*128, MSG_WAITALL);
  fprintf(stderr, "bs_conv_2_1: %ld\n", rd_size);

  conv2d(ot_maxpol_1, wt_conv_2_1, ot_conv_2_1, 128, 64, 3, 112, 112, 1, 1);
  addbias(ot_conv_2_1, ot_conv_2_1, bs_conv_2_1, 128, 112*112);
  relu(ot_conv_2_1, ot_conv_2_1, 128*112*112);

  fprintf(stderr, "conv_2_1 output: %f\n", ot_conv_2_1[0]);

  rd_size = recv(server_fd, wt_conv_2_2, sizeof(scalar_t)*128*128*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_2_2: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_2_2, sizeof(scalar_t)*128, MSG_WAITALL);
  fprintf(stderr, "bs_conv_2_2: %ld\n", rd_size);

  conv2d(ot_conv_2_1, wt_conv_2_2, ot_conv_2_2, 128, 128, 3, 112, 112, 1, 1);
  addbias(ot_conv_2_2, ot_conv_2_2, bs_conv_2_2, 128, 112*112);
  relu(ot_conv_2_2, ot_conv_2_2, 128*112*112);

  fprintf(stderr, "conv_2_2 output: %f\n", ot_conv_2_2[0]);

  maxpool2d(ot_conv_2_2, ot_maxpol_2, 128, 112, 112, 2, 2, 0);

  fprintf(stderr, "maxpool_2 output: %f\n", ot_maxpol_2[0]);


  rd_size = recv(server_fd, wt_conv_3_1, sizeof(scalar_t)*256*128*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_3_1: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_3_1, sizeof(scalar_t)*256, MSG_WAITALL);
  fprintf(stderr, "bs_conv_3_1: %ld\n", rd_size);

  conv2d(ot_maxpol_2, wt_conv_3_1, ot_conv_3_1, 256, 128, 3, 56, 56, 1, 1);
  addbias(ot_conv_3_1, ot_conv_3_1, bs_conv_3_1, 256, 56*56);
  relu(ot_conv_3_1, ot_conv_3_1, 256*56*56);

  fprintf(stderr, "conv_3_1 output: %f\n", ot_conv_3_1[0]);

  rd_size = recv(server_fd, wt_conv_3_2, sizeof(scalar_t)*256*256*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_3_2: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_3_2, sizeof(scalar_t)*256, MSG_WAITALL);
  fprintf(stderr, "bs_conv_3_2: %ld\n", rd_size);

  conv2d(ot_conv_3_1, wt_conv_3_2, ot_conv_3_2, 256, 256, 3, 56, 56, 1, 1);
  addbias(ot_conv_3_2, ot_conv_3_2, bs_conv_3_2, 256, 56*56);
  relu(ot_conv_3_2, ot_conv_3_2, 256*56*56);

  fprintf(stderr, "conv_3_2 output: %f\n", ot_conv_3_2[0]);

  rd_size = recv(server_fd, wt_conv_3_3, sizeof(scalar_t)*256*256*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_3_3: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_3_3, sizeof(scalar_t)*256, MSG_WAITALL);
  fprintf(stderr, "bs_conv_3_3: %ld\n", rd_size);

  conv2d(ot_conv_3_2, wt_conv_3_3, ot_conv_3_3, 256, 256, 3, 56, 56, 1, 1);
  addbias(ot_conv_3_3, ot_conv_3_3, bs_conv_3_3, 256, 56*56);
  relu(ot_conv_3_3, ot_conv_3_3, 256*56*56);

  fprintf(stderr, "conv_3_3 output: %f\n", ot_conv_3_3[0]);

  maxpool2d(ot_conv_3_3, ot_maxpol_3, 256, 56, 56, 2, 2, 0);

  fprintf(stderr, "maxpool_3 output: %f\n", ot_maxpol_3[0]);


  rd_size = recv(server_fd, wt_conv_4_1, sizeof(scalar_t)*512*256*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_4_1: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_4_1, sizeof(scalar_t)*512, MSG_WAITALL);
  fprintf(stderr, "bs_conv_4_1: %ld\n", rd_size);

  conv2d(ot_maxpol_3, wt_conv_4_1, ot_conv_4_1, 512, 256, 3, 28, 28, 1, 1);
  addbias(ot_conv_4_1, ot_conv_4_1, bs_conv_4_1, 512, 28*28);
  relu(ot_conv_4_1, ot_conv_4_1, 512*28*28);

  fprintf(stderr, "conv_4_1 output: %f\n", ot_conv_4_1[0]);

  rd_size = recv(server_fd, wt_conv_4_2, sizeof(scalar_t)*512*512*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_4_2: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_4_2, sizeof(scalar_t)*512, MSG_WAITALL);
  fprintf(stderr, "bs_conv_4_2: %ld\n", rd_size);

  conv2d(ot_conv_4_1, wt_conv_4_2, ot_conv_4_2, 512, 512, 3, 28, 28, 1, 1);
  addbias(ot_conv_4_2, ot_conv_4_2, bs_conv_4_2, 512, 28*28);
  relu(ot_conv_4_2, ot_conv_4_2, 512*28*28);

  fprintf(stderr, "conv_4_2 output: %f\n", ot_conv_4_2[0]);

  rd_size = recv(server_fd, wt_conv_4_3, sizeof(scalar_t)*512*512*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_4_3: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_4_3, sizeof(scalar_t)*512, MSG_WAITALL);
  fprintf(stderr, "bs_conv_4_3: %ld\n", rd_size);

  conv2d(ot_conv_4_2, wt_conv_4_3, ot_conv_4_3, 512, 512, 3, 28, 28, 1, 1);
  addbias(ot_conv_4_3, ot_conv_4_3, bs_conv_4_3, 512, 28*28);
  relu(ot_conv_4_3, ot_conv_4_3, 512*28*28);

  fprintf(stderr, "conv_4_3 output: %f\n", ot_conv_4_3[0]);

  maxpool2d(ot_conv_4_3, ot_maxpol_4, 512, 28, 28, 2, 2, 0);

  fprintf(stderr, "maxpool_4 output: %f\n", ot_maxpol_4[0]);


  rd_size = recv(server_fd, wt_conv_5_1, sizeof(scalar_t)*512*512*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_5_1: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_5_1, sizeof(scalar_t)*512, MSG_WAITALL);
  fprintf(stderr, "bs_conv_5_1: %ld\n", rd_size);

  conv2d(ot_maxpol_4, wt_conv_5_1, ot_conv_5_1, 512, 512, 3, 14, 14, 1, 1);
  addbias(ot_conv_5_1, ot_conv_5_1, bs_conv_5_1, 512, 14*14);
  relu(ot_conv_5_1, ot_conv_5_1, 512*14*14);

  fprintf(stderr, "conv_5_1 output: %f\n", ot_conv_5_1[0]);

  rd_size = recv(server_fd, wt_conv_5_2, sizeof(scalar_t)*512*512*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_5_2: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_5_2, sizeof(scalar_t)*512, MSG_WAITALL);
  fprintf(stderr, "bs_conv_5_2: %ld\n", rd_size);

  conv2d(ot_conv_5_1, wt_conv_5_2, ot_conv_5_2, 512, 512, 3, 14, 14, 1, 1);
  addbias(ot_conv_5_2, ot_conv_5_2, bs_conv_5_2, 512, 14*14);
  relu(ot_conv_5_2, ot_conv_5_2, 512*14*14);

  fprintf(stderr, "conv_5_2 output: %f\n", ot_conv_5_2[0]);

  rd_size = recv(server_fd, wt_conv_5_3, sizeof(scalar_t)*512*512*3*3, MSG_WAITALL);
  fprintf(stderr, "wt_conv_5_3: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_conv_5_3, sizeof(scalar_t)*512, MSG_WAITALL);
  fprintf(stderr, "bs_conv_5_3: %ld\n", rd_size);

  conv2d(ot_conv_5_2, wt_conv_5_3, ot_conv_5_3, 512, 512, 3, 14, 14, 1, 1);
  addbias(ot_conv_5_3, ot_conv_5_3, bs_conv_5_3, 512, 14*14);
  relu(ot_conv_5_3, ot_conv_5_3, 512*14*14);

  fprintf(stderr, "conv_5_3 output: %f\n", ot_conv_5_3[0]);

  maxpool2d(ot_conv_5_3, ot_maxpol_5, 512, 14, 14, 2, 2, 0);

  fprintf(stderr, "maxpool_5 output: %f\n", ot_maxpol_5[0]);


  rd_size = recv(server_fd, wt_connct_1, sizeof(scalar_t)*4096*7*7*512, MSG_WAITALL);
  fprintf(stderr, "wt_connct_1: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_connct_1, sizeof(scalar_t)*4096, MSG_WAITALL);
  fprintf(stderr, "bs_connct_1: %ld\n", rd_size);

  connected(ot_maxpol_5, wt_connct_1, ot_connct_1, 4096, 7*7*512);
  addbias_connected(ot_connct_1, ot_connct_1, bs_connct_1, 4096);
  relu(ot_connct_1, ot_connct_1, 4096);

  fprintf(stderr, "connected_1 output: %f\n", ot_connct_1[0]);

  // Drop out layer is skipped in inference.
  //dropout(ot_connct_1, ot_connct_1, 4096, 0.5);


  rd_size = recv(server_fd, wt_connct_2, sizeof(scalar_t)*4096*4096, MSG_WAITALL);
  fprintf(stderr, "wt_connct_2: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_connct_2, sizeof(scalar_t)*4096, MSG_WAITALL);
  fprintf(stderr, "bs_connct_2: %ld\n", rd_size);

  connected(ot_connct_1, wt_connct_2, ot_connct_2, 4096, 4096);
  addbias_connected(ot_connct_2, ot_connct_2, bs_connct_2, 4096);
  relu(ot_connct_2, ot_connct_2, 4096);

  fprintf(stderr, "connected_2 output: %f\n", ot_connct_2[0]);

  //dropout(ot_connct_2, ot_connct_2, 4096, 0.5);


  rd_size = recv(server_fd, wt_connct_3, sizeof(scalar_t)*1000*4096, MSG_WAITALL);
  fprintf(stderr, "wt_connct_3: %ld\n", rd_size);
  rd_size = recv(server_fd, bs_connct_3, sizeof(scalar_t)*1000, MSG_WAITALL);
  fprintf(stderr, "bs_connct_3: %ld\n", rd_size);

  connected(ot_connct_2, wt_connct_3, ot_connct_3, 1000, 4096);
  addbias_connected(ot_connct_3, ot_connct_3, bs_connct_3, 1000); 

  fprintf(stderr, "connected_3 output: %f\n", ot_connct_3[0]);


  softmax(ot_connct_3, ot_softmax, 1000, 1.0);

  fprintf(stderr, "softmax output: %f\n", ot_softmax[0]);

  end = clock();

  close(server_fd);

  printf("Elapsed Time: %.3f msec\n", (float)(end-start)/CLOCKS_PER_SEC*1000);

  int i;
  int topk_idx[top_K];
  top_k(ot_softmax, 1000, top_K, topk_idx);  // Extract indexes of top K possible results.
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

  // Deallocate resources.
  if (wt_conv_1_1)  free(wt_conv_1_1);
  if (bs_conv_1_1)  free(bs_conv_1_1);
  if (ot_conv_1_1)  free(ot_conv_1_1);
  if (wt_conv_1_2)  free(wt_conv_1_2);
  if (bs_conv_1_2)  free(bs_conv_1_2);
  if (ot_conv_1_2)  free(ot_conv_1_2);
  if (ot_maxpol_1)  free(ot_maxpol_1);
  if (wt_conv_2_1)  free(wt_conv_2_1);
  if (bs_conv_2_1)  free(bs_conv_2_1);
  if (ot_conv_2_1)  free(ot_conv_2_1);
  if (wt_conv_2_2)  free(wt_conv_2_2);
  if (bs_conv_2_2)  free(bs_conv_2_2);
  if (ot_conv_2_2)  free(ot_conv_2_2);
  if (ot_maxpol_2)  free(ot_maxpol_2);
  if (wt_conv_3_1)  free(wt_conv_3_1);
  if (bs_conv_3_1)  free(bs_conv_3_1);
  if (ot_conv_3_1)  free(ot_conv_3_1);
  if (wt_conv_3_2)  free(wt_conv_3_2);
  if (bs_conv_3_2)  free(bs_conv_3_2);
  if (ot_conv_3_2)  free(ot_conv_3_2);
  if (wt_conv_3_3)  free(wt_conv_3_3);
  if (bs_conv_3_3)  free(bs_conv_3_3);
  if (ot_conv_3_3)  free(ot_conv_3_3);
  if (ot_maxpol_3)  free(ot_maxpol_3);
  if (wt_conv_4_1)  free(wt_conv_4_1);
  if (bs_conv_4_1)  free(bs_conv_4_1);
  if (ot_conv_4_1)  free(ot_conv_4_1);
  if (wt_conv_4_2)  free(wt_conv_4_2);
  if (bs_conv_4_2)  free(bs_conv_4_2);
  if (ot_conv_4_2)  free(ot_conv_4_2);
  if (wt_conv_4_3)  free(wt_conv_4_3);
  if (bs_conv_4_3)  free(bs_conv_4_3);
  if (ot_conv_4_3)  free(ot_conv_4_3);
  if (ot_maxpol_4)  free(ot_maxpol_4);
  if (wt_conv_5_1)  free(wt_conv_5_1);
  if (bs_conv_5_1)  free(bs_conv_5_1);
  if (ot_conv_5_1)  free(ot_conv_5_1);
  if (wt_conv_5_2)  free(wt_conv_5_2);
  if (bs_conv_5_2)  free(bs_conv_5_2);
  if (ot_conv_5_2)  free(ot_conv_5_2);
  if (wt_conv_5_3)  free(wt_conv_5_3);
  if (bs_conv_5_3)  free(bs_conv_5_3);
  if (ot_conv_5_3)  free(ot_conv_5_3);
  if (ot_maxpol_5)  free(ot_maxpol_5);
  if (wt_connct_1)  free(wt_connct_1);
  if (bs_connct_1)  free(bs_connct_1);
  if (ot_connct_1)  free(ot_connct_1);
  if (wt_connct_2)  free(wt_connct_2);
  if (bs_connct_2)  free(bs_connct_2);
  if (ot_connct_2)  free(ot_connct_2);
  if (wt_connct_3)  free(wt_connct_3);
  if (bs_connct_3)  free(bs_connct_3);
  if (ot_connct_3)  free(ot_connct_3);
  if (ot_softmax)   free(ot_softmax);

  return 0;
}
