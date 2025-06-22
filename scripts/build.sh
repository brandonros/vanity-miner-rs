#!/bin/bash

set -e

# set architecture
VIRTUAL_ARCH=compute_120 # rtx5090 blackwell
PHYSICAL_ARCH=sm_120 # rtx5090 blackwell

# clean
cargo clean

# build kernels to get the riscv .ll file
pushd kernels
cargo build --target riscv64gc-unknown-none-elf -p kernels --release
popd

# find the riscv .ll file
RISCV_LL_FILE=$(find $CARGO_TARGET_DIR/riscv64gc-unknown-none-elf/release/deps/kernels-* -type f -name "*.ll")
if [ -z "$RISCV_LL_FILE" ]; then
    echo "No .ll file found"
    exit 1
fi

# replace uwtable attributes due to riscv core being built with unwind and not being recompiled despite panic = "abort" flag?
sed -i 's/ uwtable //g' $RISCV_LL_FILE
sed -i 's/ uwtable//g' $RISCV_LL_FILE

# transpile riscv .ll to nvptx64 .ll
pushd transpiler
cargo run --release -- $RISCV_LL_FILE
popd

# mark kernels as ptx_kernel
sed -i 's/define dso_local void @kernel_/define dso_local ptx_kernel void @kernel_/g' /tmp/output.ll

# convert the ptx .ll files to .bc files
llvm-as-19 /tmp/output.ll -o /tmp/output.bc
llvm-as-19 transpiler/assets/libintrinsics.ll -o /tmp/libintrinsics.bc

# strip debug info out of the .bc file
opt-19 -strip-debug /tmp/output.bc -o /tmp/output.bc

# compile the .bc files to .ptx
pushd nvvm_compiler
make clean
make
./build/nvvm_compiler /tmp/output.bc /tmp/libintrinsics.bc $VIRTUAL_ARCH > /tmp/output.ptx
popd

# compile the .ptx to .cubin
echo "assembling .ptx to .cubin"
ptxas -arch=$PHYSICAL_ARCH -o /tmp/output.cubin /tmp/output.ptx

# copy back
pushd nvvm_compiler
cp $RISCV_LL_FILE build/
cp /tmp/output.ll build/
cp /tmp/output.ptx build/
cp /tmp/output.cubin build/
popd
