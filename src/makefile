# Directories.
CURRDIR := $(CURDIR)
SRCDIR  := $(CURRDIR)

# Select build targets based on platform. On macOS we skip OpenMP/CUDA targets
# because the local toolchain may lack `nvcc` or OpenMP support. The full set
# will be used on Linux servers where those toolchains are expected to exist.
UNAME_S := $(shell uname -s)
TYPES := seq

ifeq ($(UNAME_S),Linux)
TYPES += omp_cpu omp_gpu cuda
else
$(warning Non-Linux detected ($(UNAME_S)); skipping OpenMP/CUDA targets. To force a full build run: `make -C $(SRCDIR) all TYPES="seq omp_cpu cuda" BINDIR=$(BINDIR)`)
endif

.PHONY: all build clean $(TYPES)

all: build

build: $(TYPES)

$(TYPES):
	@$(MAKE) -C $@ all BINDIR=$(BINDIR)

clean:
	@for d in $(TYPES); do \
		$(MAKE) -C $$d clean BINDIR=$(BINDIR); \
	done