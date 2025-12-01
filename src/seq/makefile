# Variables.
EXEC := seq
SRC := kmeans_$(EXEC).c

# Compilation flags.
CC      ?= gcc
CFLAGS  ?= -O3 -march=native
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