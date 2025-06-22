# vanity-miner-rs
GPU-accelerated vanity address generator for multiple blockchains

## How to use

```shell
cargo run --release -- solana-vanity aaa ""
cargo run --release -- ethereum-vanity 555555 ""
cargo run --release -- bitcoin-vanity bc1qqqqq ""
cargo run --release -- shallenge brandonros 000000000000cbaec87e070a04c2eb90644e16f37aab655ccdf683fdda5a6f96
```

## Apple Container

```shell
container system start
container build -t cuda-12-9-rust-builder
container run --rm -it --memory 8G -v $(pwd):/mnt cuda-12-9-rust-builder
cd /mnt
./scripts/build.sh
container system stop
```

## RISC-V Build Pipeline

* compile no_std Rust `logic` + `kernels` libraries (specifically 1.86.0 because it was built against LLVM 19) targeting riscv64gc-unknown-none-elf due to its simplicity in instruction set
* make it emit LLVM IR instead of an actual binary
* Adapt the RISC-V LLVM IR to "NVPTX64 compatiable LLVM IR"
* assemble the LLVM IR to LLVM bitcode
* feed the NVPTX64 LLVM bitcode to new CUDA toolkit 12.9 libNVVM which adds support for LLVM19 for Blackwell (previous architectures only support LLVM v7 which is very old) to get Nvidia's PTX (Parallel Thread Execution)
* feed the PTX to `ptxas` to get CUBIN SaSS (Streaming ASSembler)
* run the CUBIN on device with `gpu_runner`
