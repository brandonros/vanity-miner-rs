# Solana performance on the M5 Mac

For a setup that does not depend on the old temporary directories, follow the
[reproduction runbook](cumetal-reproduction.md). The `/tmp` commands below
describe the historical experiment; those directories were generated outputs
and binary snapshots, not repository clones.

The tested machine reports MacBook Air (Mac17,3), Apple M5, **8 GPU cores**, and
16 GB RAM. Do not substitute the 10-GPU-core M5 configuration when comparing.

The first measured 60-second run used 64 threads/block and 16 blocks: 1,024
candidate keys per launch. It completed 1,501,184 candidates in 60.00163 seconds:
**25,019 candidates/s end to end**, or **29,059 candidates/s over Metal GPU
command-buffer timestamps**. Two warmup batches and their compilation time were
excluded. Total measured GPU time was 51.65981 seconds. The CLI rate still
includes allocations, transfers, guard checks, synchronization and logging.

These are measurements of the current port, not a hardware performance ceiling.
The seven short tuning trials were sequential, not temperature-controlled or
randomized; 32 x 32 was close to 64 x 16. Treat the selected setting as a measured
starting point, not a universally optimal configuration. Power reporting during
tuning said AC Power while the battery was discharging at 54%.

| Threads/block | Blocks | Short-trial CLI candidates/s |
| ---: | ---: | ---: |
| 32 | 32 | 29,598 |
| 64 | 16 | 30,165 |
| 128 | 8 | 27,640 |
| 256 | 4 | 16,956 |
| 64 | 4 | 12,828 |
| 64 | 64 | 20,185 |
| 64 | 256 | 18,687 |

Apple recommends threadgroup sizes that are multiples of the pipeline's
[threadExecutionWidth](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth)
and within the pipeline's maximum. CuMetal requires width 32. Its runtime reads
`maxTotalThreadsPerThreadgroup` from the actual pipeline; the hardware maximum
is not a recommendation to use the largest possible block.

## Repeat the run

From `vanity-miner-cumetal`:

```sh
cargo build --release -p vanity-miner --no-default-features \
  --features cumetal,self_test,shallenge,solana,ethereum,bitcoin --locked

python3 scripts/benchmark-cumetal-solana.py \
  --cli target/release/vanity-miner \
  --library ../cuda-metal-cli/build-cli-apple/libcumetal.dylib \
  --modules /tmp/vanity-mining-final-v10 \
  --threads 64 --blocks 16 --seconds 60 \
  --out /tmp/solana-m5-60s
```

The output directory must be new. Add `--tune` to repeat the seven short trials
before the sustained run. The timer starts after two complete warmup batches,
ends at a completed batch after at least 60 seconds, and then terminates the
continuous CLI process. It does not include compilation in the timed interval.
Full CPU verification is disabled during timing. Guards and returned-match
validation remain enabled. The rare prefix is `ZZZZZZZZZZ`, suffix empty, and
batch seeds start at 1 and increment. Unexpected matches abort this fixed
benchmark fixture rather than silently introducing expensive match output.

[Report and input hashes](benchmarks/m5-solana/report.json),
[trial summary](benchmarks/m5-solana/summary.log), and
[separate full-CPU-verification check at 64 x 16](benchmarks/m5-solana/correctness-64-16.log).
Raw per-batch logs are in `/tmp/cumetal-solana-m5-benchmark`.

## Where the artifacts came from

`cuda-metal-cli/build-cli-apple` was configured as a Release CMake build with
Apple `/usr/bin/clang` and `/usr/bin/clang++`, binary shim OFF, and explicit
LZ4/Zstd include/library paths from the local Nix store. The cached configuration
is in its `CMakeCache.txt`. Rebuild that configured tree from the parent `gpu`
directory with:

```sh
cmake --build cuda-metal-cli/build-cli-apple \
  --target cumetal_runtime cumetalc -j 4
```

`cumetal_runtime` produces `libcumetal.dylib`; `cumetalc` produces the translator.
The runtime is a host library that implements the driver interface and invokes
Metal; it is not the generated mining kernel.

The `vanity-mining-final-v10` directory was made by compiling each of the four
production entries from the unchanged downloaded LLVM 21 PTX artifact using an
immutable copy of the CuMetal compiler. The historical Solana command was:

```sh
/tmp/llvm21-final-v10/build/cumetalc \
  "$HOME/Downloads/vanity-miner-aarch64-llvm21/output-llvm21.ptx" \
  --backend=cumetal-ir --ptx-strict --overwrite \
  --entry kernel_find_solana_vanity_private_key --emit=msl \
  -o /tmp/vanity-mining-final-v10/kernel_find_solana_vanity_private_key.metal
```

The other entries were `kernel_find_better_shallenge_nonce`,
`kernel_find_ethereum_vanity_private_key`, and
`kernel_find_bitcoin_vanity_private_key`. Each invocation produced `.metal` source
and its `.metal.cumetal-abi` sidecar. The CLI loads these through the runtime,
which compiles the Metal source and launches the resulting pipeline on the GPU.
Original Rust-to-PTX compilation happened separately in the CUDA/LLVM 21 build;
it does not run during this Mac benchmark.

## RTX 5090 comparison

Use the same Solana kernel/source revision, candidate generation, prefix/suffix,
seed policy, and precision/correctness requirements. Exclude compilation and
warmup on both sides. Compare candidates/second, not SHA hashes/second or match
counts. Tune CUDA block/grid sizes independently. Compare end-to-end rates with
end-to-end rates, or GPU event timings with the Metal GPU timestamp rate. This
run does not measure an RTX 5090 or justify a hardware speedup ratio. Optimized
incremental-point vanity algorithms are a different workload from deriving a
fresh Ed25519 public key for each candidate in this kernel.
