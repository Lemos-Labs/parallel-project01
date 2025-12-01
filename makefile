# Directories.
CURRDIR := $(PWD)
SRCDIR := $(CURDIR)/src
BINDIR := $(CURDIR)/bin

.PHONY: all build clean

all: build

build:
	@mkdir -p $(BINDIR)
	@$(MAKE) -C $(SRCDIR) all BINDIR=$(BINDIR)

clean:
	@rm -rf $(BINDIR)
	@$(MAKE) -C $(SRCDIR) clean BINDIR=$(BINDIR)
