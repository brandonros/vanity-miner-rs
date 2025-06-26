#!/bin/bash

set -ex

# clean
cargo clean

# build kernels to get the riscv .ll file
pushd kernels
cargo build --target riscv64gc-unknown-none-elf -p kernels --release
popd

# find the compiled riscv .ll file
RISCV_LL_FILE=$(find $CARGO_TARGET_DIR/riscv64gc-unknown-none-elf/release/deps/kernels-* -type f -name "*.ll")
if [ -z "$RISCV_LL_FILE" ]; then
    echo "No .ll file found"
    exit 1
fi

# replace uwtable attributes due to riscv core being built with unwind and not being recompiled despite panic = "abort" flag?
sed -i 's/ uwtable //g' $RISCV_LL_FILE
sed -i 's/ uwtable//g' $RISCV_LL_FILE

# transpile riscv .ll to spirv64 .ll
SPIRV_LL_FILE=/tmp/output.ll
pushd ir_adapter
cargo run --release -- spirv64 $RISCV_LL_FILE $SPIRV_LL_FILE
popd

# Convert the ptx .ll files to .bc files
llvm-as-19 $SPIRV_LL_FILE -o /tmp/output.bc

# strip debug info out of the .bc file
opt-19 -strip-debug /tmp/output.bc -o /tmp/output.bc

# Convert LLVM IR to SPIR-V using llvm-spirv
llvm-spirv \
    --spirv-target-env=SPV-IR \
    --spirv-max-version=1.6 \
    --spirv-ext-inst=none \
    --spirv-mem2reg \
    --spirv-lower-const-expr \
    -o /tmp/spirv.spv \
    /tmp/output.bc
