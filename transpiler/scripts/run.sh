#!/bin/bash

. ~/.bash_profile

# clean
cargo clean

# build kernels
pushd ../kernels
cargo build --release || exit 1
popd

# find the .ll file
LL_FILE=$(find $CARGO_TARGET_DIR/riscv64gc-unknown-none-elf/release/deps/kernels-* -type f -name "*.ll")
if [ -z "$LL_FILE" ]; then
    echo "No .ll file found"
    exit 1
fi

# generate the .ll file
cargo run --release -- $LL_FILE || exit 1

# convert the .ll files to .bc files
llvm-as /tmp/output.ll -o /tmp/output.bc
llvm-as assets/libintrinsics.ll -o /tmp/libintrinsics.bc

# copy the .bc files to the nvvm_compiler directory
scp /tmp/output.bc brandon@asusrogstrix.local:
scp /tmp/libintrinsics.bc brandon@asusrogstrix.local:

# compile the .bc files
ssh brandon@asusrogstrix.local "cd /home/brandon/nvvm_compiler && ./compile.sh && ./nvvm_compiler output.bc libintrinsics.bc > output.ptx" || exit 1
ssh brandon@asusrogstrix.local "cd /home/brandon/nvvm_compiler && ptxas -arch=sm_120 -o output.cubin output.ptx" || exit 1
ssh brandon@asusrogstrix.local "cd /home/brandon/nvvm_compiler && nvcc -fatbin -arch=sm_120 -o output.fatbin output.ptx" || exit 1
