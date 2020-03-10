#include "server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int init_server(struct sockaddr_in* server_addr, int* server_fd, int port)
{
  if ((*server_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    return SERVER_FAIL;
  }
  memset(server_addr, 0x00, sizeof(struct sockaddr_in));

  server_addr->sin_family = AF_INET;
  server_addr->sin_addr.s_addr = htonl(INADDR_ANY);
  server_addr->sin_port = htons(port);

  return SERVER_SUCCESS;
}
