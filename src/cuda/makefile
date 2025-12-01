# Variables.
EXEC   := cuda
SRC    := kmeans_$(EXEC).cu

# Compilation flags.
NVCC    ?= nvcc
NVFLAGS ?= -O3 -arch=sm_61 -std=c++11

TARGET := $(BINDIR)/kmeans_$(EXEC)

.PHONY: all build clean

all: build

build: $(NVCC) $(NVFLAGS) -o $(TARGET) kmeans_cuda.cu

