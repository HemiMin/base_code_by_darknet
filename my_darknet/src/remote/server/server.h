#ifndef SERVER_H_
#define SERVER_H_

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define SERVER_FAIL 0
#define SERVER_SUCCESS 1

int init_server(struct sockaddr_in* server_addr, int* server_fd, int port);

#endif
