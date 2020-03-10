#include "client.h"

#include <string.h>

int init_client(struct sockaddr_in* server_addr, int* server_fd, char* server_ip, int server_port)
{
  if ((*server_fd = socket(PF_INET, SOCK_STREAM, 0)) < 0) {
    return CLIENT_FAIL;
  }
  memset(server_addr, 0x00, sizeof(struct sockaddr_in));

  server_addr->sin_family = AF_INET;
  server_addr->sin_addr.s_addr = inet_addr(server_ip);
  server_addr->sin_port = htons(server_port);

  return CLIENT_SUCCESS;
}
