#!/bin/bash

cc main.c -I/usr/local/cuda-12.9/nvvm/include -L/usr/local/cuda-12.9/nvvm/lib64 -lnvvm -o nvvm_compiler
