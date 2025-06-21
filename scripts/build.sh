#!/bin/bash

set -e

# set architecture
VIRTUAL_ARCH=compute_120 # rtx5090 blackwell
PHYSICAL_ARCH=sm_120 # rtx5090 blackwell

# clean
cargo clean

# build kernels to get the .ll file
pushd kernels
cargo build --target riscv64gc-unknown-none-elf --release
popd

# find the .ll file
LL_FILE=$(find $CARGO_TARGET_DIR/riscv64gc-unknown-none-elf/release/deps/kernels-* -type f -name "*.ll")
if [ -z "$LL_FILE" ]; then
    echo "No .ll file found"
    exit 1
fi

# replace uwtable attributes due to riscv core being built with unwind and not being recompiled despite panic = "abort" flag?
sed -i 's/ uwtable //g' $LL_FILE
sed -i 's/ uwtable//g' $LL_FILE

# transpile riscv .ll to nvptx64 .ll
pushd transpiler
cargo run --release -- $LL_FILE
popd

# convert the .ll files to .bc files
llvm-as-19 /tmp/output.ll -o /tmp/output.bc
llvm-as-19 transpiler/assets/libintrinsics.ll -o /tmp/libintrinsics.bc

# compile the .bc files to .ptx
pushd nvvm_compiler
make
./build/nvvm_compiler /tmp/output.bc /tmp/libintrinsics.bc $VIRTUAL_ARCH > /tmp/output.ptx
popd

# compile the .ptx to .cubin and .fatbin
echo "assembling .ptx to .cubin"
ptxas -arch=$PHYSICAL_ARCH -o /tmp/output.cubin /tmp/output.ptx
echo "assembling .ptx to .fatbin"
nvcc -fatbin -arch=$PHYSICAL_ARCH -o /tmp/output.fatbin /tmp/output.ptx

# copy back
pushd nvvm_compiler
cp /tmp/output.ptx build/
cp /tmp/output.cubin build/
cp /tmp/output.fatbin build/
popd
