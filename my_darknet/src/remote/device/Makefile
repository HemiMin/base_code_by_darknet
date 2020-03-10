CC=gcc

TARGET=vgg16

CFLAGS=-std=c99 -Ofast -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC
LDFLAGS=-lm
CSRCS=conv.c connected.c maxpool.c addbias.c relu.c stb_image.c dropout.c softmax.c imagenet_classifier.c topk.c crop.c darknet_classifier.c client.c
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
