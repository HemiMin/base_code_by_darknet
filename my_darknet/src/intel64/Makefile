CC=gcc
MKL=0
PLANNER=1
#1: vgg16  2: resnet50  3: alexnet  4: squeezenet
DNN=1
TIME_ESTIMATE=0

ifeq ($(DNN), 1)
TARGET=vgg16
endif
ifeq ($(DNN), 2)
TARGET=resnet50
endif
ifeq ($(DNN), 3)
TARGET=alexnet
endif
ifeq ($(DNN), 4)
TARGET=squeezenet
endif

CFLAGS=-std=c99 -Ofast -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC
LDFLAGS=-lm
CDEF=
ifeq ($(MKL), 1)
LDFLAGS+= -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
CDEF+= -DMKL
endif
ifeq ($(TIME_ESTIMATE), 1)
CDEF+= -DTIME_ESTIMATE
endif
ifeq ($(PLANNER), 1)
LAYER_SRCS_DIR=$(TARGET).srcs
CDEF+= -DPLANNER
endif
CSRCS=conv.c connected.c maxpool.c batchnorm.c addbias.c relu.c stb_image.c dropout.c softmax.c imagenet_classifier.c topk.c crop.c darknet_classifier.c residual.c route.c leaky_relu.c avgpool.c
ifeq ($(PLANNER), 1)
CSRCS+= scat_gather.c
CSRCS+= $(wildcard $(LAYER_SRCS_DIR)/*.c)
endif
CSRCS+= $(TARGET).c
COBJS=$(CSRCS:.c=.o)

INCLUDES=-I.

all: $(COBJS)
	$(CC) $(COBJS) $(LDFLAGS) -o $(TARGET)

.c.o:
	$(CC) $(CFLAGS) $(CDEF) $(LDFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(COBJS) $(TARGET)
