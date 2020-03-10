CC=gcc
CXX=g++
NVCC=nvcc
CUDA=
CUBLAS=
OPENBLAS=0
PLANNER=1

rw=
net=
version=

#1: vgg16  2: resnet50  3: alexnet  4: squeezenet
ifeq ($(net), vgg16)
DNN=1
endif
ifeq ($(net), resnet50)
DNN=2
endif
ifeq ($(net), alexnet)
DNN=3
endif
ifeq ($(net), squeezenet)
DNN=4
endif
ifeq ($(net), yolov2)
DNN=5
endif

TIME_ESTIMATE=
OUTPUT_NAME=

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
ifeq ($(DNN), 5)
TARGET=yolov2
endif

TARGET_SRC=$(TARGET).$(hw).$(version)
OUTPUT_NAME=$(TARGET_SRC)
ifeq ($(TIME_ESTIMATE), 1)
OUTPUT_NAME=$(TARGET_SRC).timeest
endif

FLAGS=-Ofast -Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -fPIC
CUFLAGS=-std=c++11 $(FLAGS)
CFLAGS=-std=c99 $(FLAGS)
CDEF=
LDFLAGS=-lm
LDPATH=
INCLUDES=-I.
ifeq ($(CUDA), 1)
CDEF+= -DCUDA
LDFLAGS+= -lcudart
LDPATH+= -L/usr/local/cuda/lib64
endif
ifeq ($(CUBLAS), 1)
CDEF+= -DCUBLAS
LDFLAGS+= -lcublas -lcudart
LDPATH+= -L/usr/local/cuda/lib64
else ifeq ($(OPENBLAS), 1)
CDEF+= -DOPENBLAS
LDFLAGS+= -lopenblas
LDPATH+= -L/opt/OpenBLAS/lib
INCLUDES+= -I/opt/OpenBLAS/include
endif
ifeq ($(TIME_ESTIMATE), 1)
CDEF+= -DTIME_ESTIMATE
endif
ifeq ($(PLANNER), 1)
LAYER_SRCS_DIR=$(TARGET_SRC).srcs
CDEF+= -DPLANNER
endif
CSRCS=conv.c connected.c maxpool.c batchnorm.c addbias.c relu.c stb_image.c dropout.c softmax.c imagenet_classifier.c topk.c crop.c darknet_classifier.c residual.c route.c leaky_relu.c logistic.c avgpool.c reorg.c region.c coco_classifier.c
CUSRCS=
ifeq ($(CUDA), 1)
CUSRCS+= conv_gpu.cu timer_gpu.cu malloc_gpu.cu
endif
ifeq ($(CUBLAS), 1)
CUSRCS+= conv_gpu.cu timer_gpu.cu malloc_gpu.cu
endif

ifeq ($(PLANNER), 1)
	ifeq ($(CUDA), 1)
	CUSRCS+= scat_gather_gpu.cu
	else ifeq ($(CUBLAS), 1)
	CUSRCS+= scat_gather_gpu.cu
	else
	CSRCS+= scat_gather.c
	endif
CSRCS+= $(wildcard $(LAYER_SRCS_DIR)/*.c)
endif
CSRCS+= $(TARGET).$(hw).c
COBJS=$(CSRCS:.c=.o)
ifeq ($(CUDA), 1)
COBJS+=$(CUSRCS:.cu=.o)
endif
ifeq ($(CUBLAS), 1)
COBJS+=$(CUSRCS:.cu=.o)
endif

all: $(COBJS)
  ifeq ($(CUDA), 1)
	$(CXX) $(COBJS) $(LDPATH) $(LDFLAGS) -o $(OUTPUT_NAME).cuda
  endif
  ifeq ($(CUBLAS), 1)
	$(CXX) $(COBJS) $(LDPATH) $(LDFLAGS) -o $(OUTPUT_NAME)
  endif

%.o: %.c
	$(CC) $(CFLAGS) $(CDEF) $(LDFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu
	$(NVCC) --compiler-options "$(CUFLAGS)" $(CDEF) $(LDFLAGS) -c $< -o $@

clean:
	rm -f $(COBJS) $(OUTPUT_NAME)
