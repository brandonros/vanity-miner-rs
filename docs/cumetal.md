# Running prebuilt Rust-CUDA PTX through CuMetal

The optional `cumetal` CLI backend consumes an existing PTX artifact on Apple
Silicon. It does not invoke Rust-CUDA or require the NVIDIA SDK on the Mac.
The CuMetal compiler and runtime must already be built. This flow asks
`cumetalc` to emit Metal source, then the runtime compiles it through the Metal
API. It was validated on Apple M5 with the standalone `metal` and `metallib`
commands unavailable to `xcrun`; those commands are not used by this flow.
Building CuMetal itself and using its other compilation paths have separate
toolchain requirements.
The `gpu`/`llvm21` features select the NVIDIA backend and cannot be combined
with `cumetal`.

Build all CLI modes:

```sh
cargo build -p vanity-miner --no-default-features \
  --features cumetal,self_test,shallenge,solana,ethereum,bitcoin --locked
```

Start with a bounded CPU-verified Shallenge run:

```sh
CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0 \
CUMETAL_TRACE_GPU=1 \
./target/debug/vanity-miner \
  --cumetal-library /path/to/cuda-metal/build/libcumetal.dylib \
  --cumetalc /path/to/cuda-metal/build/cumetalc \
  --ptx /path/to/output-llvm21.ptx \
  --batches 1 --threads-per-block 1 --blocks 1 --seed 1 --verify \
  shallenge brandonros ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
```

For a bounded Solana run, replace the final command and its positional arguments
with `solana-vanity '' ''`. An empty prefix and suffix deliberately match every
candidate, making it useful for validation. `--verify` computes every candidate
on CPU and checks the match count; the returned candidate is compared against
CPU even without this flag. Input buffers and output guards are checked on every
launch. CPU verification is for correctness testing and reduces throughput.

Omitting `--batches` continues searching. With `--seed`, each batch increments the
seed with wrapping arithmetic; otherwise batches use random seeds. Thread and
block counts control the launch size. Start small while validating a new artifact.

## Compilation and reuse

`--ptx` compiles the selected entry with strict typed CuMetal lowering. The CLI
reads the original artifact without changing it. First-time PTX and Apple Metal
compilation can take minutes for large kernels; this is distinct from kernel
execution time.

The source cache defaults to `.cumetal-cache` (override with `--cumetal-cache`).
Its key includes PTX bytes, compiler bytes, the selected entry, and compiler
options. Metal source and its ABI sidecar are published together after a
successful build. Failed builds are not cached. CuMetal owns the later Metal
compilation cache.

Alternatively, supply `--module-dir /path/to/modules`. It must contain
`ENTRY.metal` and `ENTRY.metal.cumetal-abi` for each requested entry. Use sidecars
from the same compiler run as the source; guessing pointer/scalar arguments can
make a launch incorrect even when Metal compilation succeeds.

## Self-tests

Use the same library/artifact options with `self-test`. The CLI runs the launch
probe and all 118 numerical entries, checking each selected slot equals 1 while
all other slots and guards remain intact. Repeat `--self-test-slot N` to run a
subset while debugging. Each selected entry has its own compiled module; a full
uncached run can be lengthy.

Once all self-test modules and their sidecars are compiled, capture a complete
CLI run with:

```sh
python3 scripts/validate-cumetal-self-tests.py \
  --cli target/debug/vanity-miner \
  --library /path/to/cuda-metal/build/libcumetal.dylib \
  --modules /path/to/self-test-modules \
  --out /path/to/new-self-test-report
```

This wrapper snapshots the CLI, runtime, kernel source inventory, and all 119
modules with their sidecars. It requires exactly one numerical pass and one
successful GPU launch for each entry, including the probe; missing or duplicate
results fail the report. It refuses to start if any module is missing. The
report's `complete` flag means the attempt finished; only `passed: true` means
the complete inventory passed. Allow space for a second copy of the modules.

## Current validation limits

Run the reproducible bounded CLI matrix against a directory containing all four
compiled mining modules and their ABI sidecars:

```sh
python3 scripts/validate-cumetal-cli.py \
  --cli target/debug/vanity-miner \
  --library /path/to/cuda-metal/build/libcumetal.dylib \
  --modules /path/to/modules \
  --out /path/to/new-validation-report
```

The runner snapshots the executable, runtime, and modules, records hashes, and
checks nine scenarios per mode. It covers matching/nonmatching inputs, launch
boundaries, multiple blocks, repeated batches, and wrapping seeds. Each launch
uses CPU verification and buffer guards. The output directory must be new;
failures and timeouts remain in the report. Use `--mode` to select one mining
mode. This matrix does not replace the separate 118-entry self-test run.

This backend is experimental. On Apple M5, the pinned LLVM 21 candidate passed
all 36 bounded CLI scenarios across Shallenge, Solana, Ethereum, and Bitcoin
(69 CPU-verified batches), including independent known-positive requests.
The Ethereum request regression checks hex decoding against a fixed expected
address, because CPU/GPU agreement alone can miss identical host-input errors.

The same pinned compiler passed all 118 Rust-generated numerical entries and
the separate launch probe through the standalone GPU runner. Seven recovered
slots plus the probe also passed through the CLI. The full CLI self-test wrapper
above has host tests for incomplete/duplicate evidence; its entire 119-entry GPU
run has not yet been validated. These results do not claim a complete run on
newer compiler binaries.

Validation inputs:

- Rust source: `6a5b9e6` plus this CLI integration; kernel sources are unchanged.
- LLVM 21 PTX SHA-256: `2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b`.
- Pinned compiler SHA-256: `f60d200915889cebda78be4692905ec899a66bd4348fbc75960d6b5ce909837f`.
- Validated CLI SHA-256: `df5d6c461ac04fb12d0f2a1aa3cfd29fdcda2995ea57210e8a9c0678db4d6d19`.

Further Rust-to-GPU code-generation coverage belongs in the Rust kernel suite:
runtime-supplied 64-bit values, carry/wrap and shift boundaries, loop-carried
state, indexed tables, and scalar/pointer use across branches. Compile those
Rust kernels through Rust-CUDA and run the resulting artifact on the GPU;
CLI host tests and handwritten PTX regressions do not substitute for that path.
See [coverage issue #16](https://github.com/brandonros/vanity-miner-rs/issues/16).
