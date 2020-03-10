#ifndef CLIENT_H_
#define CLIENT_H_

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define CLIENT_FAIL 0
#define CLIENT_SUCCESS 1

int init_client(struct sockaddr_in* server_addr, int* server_fd, char* server_ip, int server_port);

#endif
