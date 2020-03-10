#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "type.h"
#include "server.h"

int main(int argc, char* argv[])
{
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <weights file path>.weights <server port> <num client>\n", argv[0]);
    exit(0);
  }

  FILE* wt_fp = fopen(argv[1], "rb");
  if (wt_fp == NULL) {
    fprintf(stderr, "File %s is not opened.\n", argv[2]);
    exit(0);
  }

  // Memory Allocation
  scalar_t* wt_conv_1_1 = (scalar_t*)calloc(3*3*3*64, sizeof(scalar_t));
  scalar_t* bs_conv_1_1 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* wt_conv_1_2 = (scalar_t*)calloc(3*3*64*64, sizeof(scalar_t));
  scalar_t* bs_conv_1_2 = (scalar_t*)calloc(64, sizeof(scalar_t));
  scalar_t* wt_conv_2_1 = (scalar_t*)calloc(3*3*64*128, sizeof(scalar_t));
  scalar_t* bs_conv_2_1 = (scalar_t*)calloc(128, sizeof(scalar_t));
  scalar_t* wt_conv_2_2 = (scalar_t*)calloc(3*3*128*128, sizeof(scalar_t));
  scalar_t* bs_conv_2_2 = (scalar_t*)calloc(128, sizeof(scalar_t));
  scalar_t* wt_conv_3_1 = (scalar_t*)calloc(3*3*128*256, sizeof(scalar_t));
  scalar_t* bs_conv_3_1 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* wt_conv_3_2 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  scalar_t* bs_conv_3_2 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* wt_conv_3_3 = (scalar_t*)calloc(3*3*256*256, sizeof(scalar_t));
  scalar_t* bs_conv_3_3 = (scalar_t*)calloc(256, sizeof(scalar_t));
  scalar_t* wt_conv_4_1 = (scalar_t*)calloc(3*3*256*512, sizeof(scalar_t));
  scalar_t* bs_conv_4_1 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* wt_conv_4_2 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_4_2 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* wt_conv_4_3 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_4_3 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* wt_conv_5_1 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_5_1 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* wt_conv_5_2 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_5_2 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* wt_conv_5_3 = (scalar_t*)calloc(3*3*512*512, sizeof(scalar_t));
  scalar_t* bs_conv_5_3 = (scalar_t*)calloc(512, sizeof(scalar_t));
  scalar_t* wt_connct_1 = (scalar_t*)calloc(7*7*512*4096, sizeof(scalar_t));
  scalar_t* bs_connct_1 = (scalar_t*)calloc(4096, sizeof(scalar_t));
  scalar_t* wt_connct_2 = (scalar_t*)calloc(4096*4096, sizeof(scalar_t));
  scalar_t* bs_connct_2 = (scalar_t*)calloc(4096, sizeof(scalar_t));
  scalar_t* wt_connct_3 = (scalar_t*)calloc(4096*1000, sizeof(scalar_t));
  scalar_t* bs_connct_3 = (scalar_t*)calloc(1000, sizeof(scalar_t));

  // Read weights and bias data from file
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
  fread((char*)wt_connct_1, sizeof(scalar_t), 7*7*512*4096, wt_fp);
  fread((char*)bs_connct_1, sizeof(scalar_t), 4096, wt_fp);
  fread((char*)wt_connct_2, sizeof(scalar_t), 4096*4096, wt_fp);
  fread((char*)bs_connct_2, sizeof(scalar_t), 4096, wt_fp);
  fread((char*)wt_connct_3, sizeof(scalar_t), 4096*1000, wt_fp);
  fread((char*)bs_connct_3, sizeof(scalar_t), 1000, wt_fp);

  fclose(wt_fp);

  // Network setting.
  int port = atoi(argv[2]);
  int num_client = atoi(argv[3]);
  struct sockaddr_in server_addr, client_addr;
  int server_fd, client_fd;
  socklen_t len;
  char client_ip[20];

  if (init_server(&server_addr, &server_fd, port) == SERVER_FAIL) {
    fprintf(stderr, "Server: Cannot open stream socket.\n");
    exit(0);
  }

  if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
    fprintf(stderr, "Server: Cannot bind local address.\n");
    exit(0);
  }

  if (listen(server_fd, num_client) < 0) {
    fprintf(stderr, "Server: Cannot listening connect.\n");
    exit(0);
  }
  fprintf(stderr, "Server : Waiting connection request.\n");

  len = (socklen_t)sizeof(client_addr);
  client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &len);
  if (client_fd < 0) {
    fprintf(stderr, "Server : accept failed.\n");
    exit(0);
  }
  inet_ntop(AF_INET, &client_addr.sin_addr.s_addr, client_ip, sizeof(client_ip));
  fprintf(stderr, "Server : %s client connected.\n", client_ip);

  size_t wr_size;
  clock_t start, end;
  start = clock();

  wr_size = send(client_fd, wt_conv_1_1, sizeof(scalar_t)*64*3*3*3, 0);
  fprintf(stderr, "wt_conv_1_1: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_1_1, sizeof(scalar_t)*64, 0);
  fprintf(stderr, "bs_conv_1_1: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_1_2, sizeof(scalar_t)*64*64*3*3, 0);
  fprintf(stderr, "wt_conv_1_2: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_1_2, sizeof(scalar_t)*64, 0);
  fprintf(stderr, "bs_conv_1_2: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_2_1, sizeof(scalar_t)*128*64*3*3, 0);
  fprintf(stderr, "wt_conv_2_1: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_2_1, sizeof(scalar_t)*128, 0);
  fprintf(stderr, "bS_conv_2_1: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_2_2, sizeof(scalar_t)*128*128*3*3, 0);
  fprintf(stderr, "wt_conv_2_2: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_2_2, sizeof(scalar_t)*128, 0);
  fprintf(stderr, "bs_conv_2_2: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_3_1, sizeof(scalar_t)*256*128*3*3, 0);
  fprintf(stderr, "wt_conv_3_1: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_3_1, sizeof(scalar_t)*256, 0);
  fprintf(stderr, "bs_conv_3_1: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_3_2, sizeof(scalar_t)*256*256*3*3, 0);
  fprintf(stderr, "wt_conv_3_2: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_3_2, sizeof(scalar_t)*256, 0);
  fprintf(stderr, "bs_conv_3_2: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_3_3, sizeof(scalar_t)*256*256*3*3, 0);
  fprintf(stderr, "wt_conv_3_3: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_3_3, sizeof(scalar_t)*256, 0);
  fprintf(stderr, "bs_conv_3_3: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_4_1, sizeof(scalar_t)*512*256*3*3, 0);
  fprintf(stderr, "wt_conv_4_1: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_4_1, sizeof(scalar_t)*512, 0);
  fprintf(stderr, "bs_conv_4_1: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_4_2, sizeof(scalar_t)*512*512*3*3, 0);
  fprintf(stderr, "wt_conv_4_2: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_4_2, sizeof(scalar_t)*512, 0);
  fprintf(stderr, "bs_conv_4_2: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_4_3, sizeof(scalar_t)*512*512*3*3, 0);
  fprintf(stderr, "wt_conv_4_3: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_4_3, sizeof(scalar_t)*512, 0);
  fprintf(stderr, "bs_conv_4_3: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_5_1, sizeof(scalar_t)*512*512*3*3, 0);
  fprintf(stderr, "wt_conv_5_1: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_5_1, sizeof(scalar_t)*512, 0);
  fprintf(stderr, "bs_conv_5_1: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_5_2, sizeof(scalar_t)*512*512*3*3, 0);
  fprintf(stderr, "wt_conv_5_2: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_5_2, sizeof(scalar_t)*512, 0);
  fprintf(stderr, "bs_conv_5_2: %ld\n", wr_size);

  wr_size = send(client_fd, wt_conv_5_3, sizeof(scalar_t)*512*512*3*3, 0);
  fprintf(stderr, "wt_conv_5_3: %ld\n", wr_size);
  wr_size = send(client_fd, bs_conv_5_3, sizeof(scalar_t)*512, 0);
  fprintf(stderr, "bs_conv_5_3: %ld\n", wr_size);

  wr_size = send(client_fd, wt_connct_1, sizeof(scalar_t)*4096*7*7*512, 0);
  fprintf(stderr, "wt_connct_1: %ld\n", wr_size);
  wr_size = send(client_fd, bs_connct_1, sizeof(scalar_t)*4096, 0);
  fprintf(stderr, "bs_conncT_1: %ld\n", wr_size);

  wr_size = send(client_fd, wt_connct_2, sizeof(scalar_t)*4096*4096, 0);
  fprintf(stderr, "wt_connct_2: %ld\n", wr_size);
  wr_size = send(client_fd, bs_connct_2, sizeof(scalar_t)*4096, 0);
  fprintf(stderr, "bs_connct_2: %ld\n", wr_size);

  wr_size = send(client_fd, wt_connct_3, sizeof(scalar_t)*1000*4096, 0);
  fprintf(stderr, "wt_connct_3: %ld\n", wr_size);
  wr_size = send(client_fd, bs_connct_3, sizeof(scalar_t)*1000, 0);
  fprintf(stderr, "bs_conncT_3: %ld\n", wr_size);

  end = clock();

  printf("Elapsed Server Time: %.3f msec\n", (float)(end-start)/CLOCKS_PER_SEC*1000);

  close(client_fd);
  close(server_fd);

  // Deallocate resources.
  if (wt_conv_1_1)  free(wt_conv_1_1);
  if (bs_conv_1_1)  free(bs_conv_1_1);
  if (wt_conv_1_2)  free(wt_conv_1_2);
  if (bs_conv_1_2)  free(bs_conv_1_2);
  if (wt_conv_2_1)  free(wt_conv_2_1);
  if (bs_conv_2_1)  free(bs_conv_2_1);
  if (wt_conv_2_2)  free(wt_conv_2_2);
  if (bs_conv_2_2)  free(bs_conv_2_2);
  if (wt_conv_3_1)  free(wt_conv_3_1);
  if (bs_conv_3_1)  free(bs_conv_3_1);
  if (wt_conv_3_2)  free(wt_conv_3_2);
  if (bs_conv_3_2)  free(bs_conv_3_2);
  if (wt_conv_3_3)  free(wt_conv_3_3);
  if (bs_conv_3_3)  free(bs_conv_3_3);
  if (wt_conv_4_1)  free(wt_conv_4_1);
  if (bs_conv_4_1)  free(bs_conv_4_1);
  if (wt_conv_4_2)  free(wt_conv_4_2);
  if (bs_conv_4_2)  free(bs_conv_4_2);
  if (wt_conv_4_3)  free(wt_conv_4_3);
  if (bs_conv_4_3)  free(bs_conv_4_3);
  if (wt_conv_5_1)  free(wt_conv_5_1);
  if (bs_conv_5_1)  free(bs_conv_5_1);
  if (wt_conv_5_2)  free(wt_conv_5_2);
  if (bs_conv_5_2)  free(bs_conv_5_2);
  if (wt_conv_5_3)  free(wt_conv_5_3);
  if (bs_conv_5_3)  free(bs_conv_5_3);
  if (wt_connct_1)  free(wt_connct_1);
  if (bs_connct_1)  free(bs_connct_1);
  if (wt_connct_2)  free(wt_connct_2);
  if (bs_connct_2)  free(bs_connct_2);
  if (wt_connct_3)  free(wt_connct_3);
  if (bs_connct_3)  free(bs_connct_3);

  return 0;
}
