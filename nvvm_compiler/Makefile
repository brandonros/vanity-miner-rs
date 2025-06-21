# Simple Makefile for NVVM compiler

CC = cc
CUDA_PATH = /usr/local/cuda-12.9
CFLAGS = -I$(CUDA_PATH)/nvvm/include
LDFLAGS = -L$(CUDA_PATH)/nvvm/lib64 -lnvvm
BUILD_DIR = build
TARGET = $(BUILD_DIR)/nvvm_compiler
SOURCE = main.c

$(TARGET): $(SOURCE) | $(BUILD_DIR)
	$(CC) $(SOURCE) $(CFLAGS) $(LDFLAGS) -o $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: clean