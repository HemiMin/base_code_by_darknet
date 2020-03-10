CC=gcc

TARGET=vgg16

CFLAGS=-std=c99 -Ofast -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC
LDFLAGS=
CSRCS=server.c
CSRCS+=vgg16.c
COBJS=$(CSRCS:.c=.o)
CDEF=

INCLUDES=-I.

all: $(COBJS)
	$(CC) $(COBJS) $(LDFLAGS) -o $(TARGET)

.c.o:
	$(CC) $(CFLAGS) $(CDEF) $(LDFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(COBJS) $(TARGET)
