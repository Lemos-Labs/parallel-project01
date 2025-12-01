EXEC   := omp_gpu
SRC    := kmeans_$(EXEC).c

CC      ?= clang
CFLAGS  ?= -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
LDFLAGS ?= -lm

TARGET := $(BINDIR)/kmeans_$(EXEC)

.PHONY: all build clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	@rm -f $(TARGET)
