# Directories.
CURRDIR := $(CURDIR)
SRCDIR  := $(CURRDIR)

# Variables.
TYPES := seq omp_cpu cuda

.PHONY: all build clean $(TYPES)

all: build

build: $(TYPES)

$(TYPES):
	@$(MAKE) -C $@ all BINDIR=$(BINDIR)

clean:
	@for d in $(TYPES); do \
		$(MAKE) -C $$d clean BINDIR=$(BINDIR); \
	done