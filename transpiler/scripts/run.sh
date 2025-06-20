#!/bin/bash

. ~/.bash_profile

# generate the .ll file
cargo run || exit 1

# convert the .ll files to .bc files
llvm-as /tmp/output.ll -o /tmp/output.bc
llvm-as assets/libintrinsics.ll -o /tmp/libintrinsics.bc

# copy the .bc files to the nvvm_compiler directory
scp /tmp/output.bc brandon@asusrogstrix.local:
scp /tmp/libintrinsics.bc brandon@asusrogstrix.local:

# compile the .bc files
ssh brandon@asusrogstrix.local "cd /home/brandon/nvvm_compiler && ./compile.sh && ./nvvm_compiler output.bc libintrinsics.bc > output.ptx"
ssh brandon@asusrogstrix.local "cd /home/brandon/nvvm_compiler && ptxas -arch=sm_120 -o output.cubin output.ptx"
ssh brandon@asusrogstrix.local "cd /home/brandon/nvvm_compiler && nvcc -fatbin -arch=sm_120 -o output.fatbin output.ptx"
