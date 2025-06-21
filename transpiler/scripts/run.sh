#!/bin/bash

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

# replace uwtable attributes with no_uwtable attributes due to riscv core being built with unwind?
sed -i 's/ uwtable //g' $LL_FILE
sed -i 's/ uwtable//g' $LL_FILE

# generate the .ll file
cargo run --release -- $LL_FILE || exit 1

# convert the .ll files to .bc files
llvm-as-19 /tmp/output.ll -o /tmp/output.bc || exit 1
llvm-as-19 assets/libintrinsics.ll -o /tmp/libintrinsics.bc || exit 1

# compile the .bc files
pushd ../nvvm_compiler
make || exit 1
./build/nvvm_compiler /tmp/output.bc /tmp/libintrinsics.bc > /tmp/output.ptx || exit 1
ptxas -arch=sm_120 -o /tmp/output.cubin /tmp/output.ptx || exit 1
nvcc -fatbin -arch=sm_120 -o /tmp/output.fatbin /tmp/output.ptx || exit 1
popd
