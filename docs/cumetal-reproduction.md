# Reproduce the Rust-CUDA → CuMetal → M5 workflow

This is the handoff/runbook for the `poc/cumetal-cli` branch of vanity-miner-rs
([PR #18](https://github.com/brandonros/vanity-miner-rs/pull/18)). It explains the
inputs, build steps, generated files, correctness checks, and benchmark. Commands
below use workspace directories, not the historical `/tmp` snapshots.

## 1. What runs where

```text
vanity-miner Rust kernels + logic
    │ Rust-CUDA / CUDA / LLVM 21 in Linux GitHub Actions
    ▼
output-llvm21.ptx                     one PTX module, many entrypoints
    │ cumetalc on macOS, once per selected entrypoint
    ▼
ENTRY.metal + ENTRY.metal.cumetal-abi  generated source + argument layout
    │ vanity-miner CLI loads libcumetal.dylib
    ▼
Apple Metal compiles the source, then launches it on the M5 GPU
```

- **vanity-miner-rs** contains the application, shared CPU/kernel logic, and Rust
  kernel entrypoints. The Mac executable is `target/release/vanity-miner` built
  with the `cumetal` feature.
- **Rust-CUDA** compiles the Rust kernels to NVIDIA PTX. For this workflow that
  happens in CI, not on the Mac. Its LLVM 21 dependency is part of that build.
- **CuMetal**, from [Lulzx/cuda-metal](https://github.com/Lulzx/cuda-metal), is
  the PTX translator and Metal-backed runtime. `cumetalc` translates PTX;
  `libcumetal.dylib` supplies the CUDA-style driver interface the CLI loads.
  You do not start CuMetal as a server or run its CLI to search for addresses.
- The Linux executable bundled beside the PTX in the CI artifact is not the Mac
  CLI. Build the Mac CLI locally; reuse only the PTX from that artifact.

The artifact's `aarch64` label identifies the Linux build host, not an Apple
GPU target. The downloaded PTX declares PTX 9.3 / `sm_100`; CuMetal consumes that
PTX and produces Metal source.

## 2. Worktrees and the misleading temporary names

On the machine used for validation:

| Directory | Role |
| --- | --- |
| `vanity-miner-cumetal` | vanity-miner-rs worktree, branch `poc/cumetal-cli` |
| `cuda-metal-cli` | CuMetal worktree, branch `poc/vanity-cli-validation` |
| `rust-cuda` and its other worktrees | Compiler development; not required to run downloaded PTX on the Mac |
| `/tmp/llvm21-final-v10/build` | Historical copies of `cumetalc` and `libcumetal.dylib` to freeze a test run |
| `/tmp/llvm21-final-v10` | Historical self-test modules, logs, and reports |
| `/tmp/vanity-mining-final-v10` | Historical generated modules for the four mining entrypoints |

**`llvm21-final-v10` was not a repository clone.** `v10` meant iteration 10 of
our local validation candidate. It was not an LLVM version, dependency name,
or directory that a new setup must obtain. The binaries had been built from
`cuda-metal-cli` and copied there so another agent's rebuild could not change a
running test. This guide builds them from source instead.

Before using an existing workspace, inspect it rather than changing its branches:

```sh
git worktree list
git -C ../cuda-metal-cli worktree list
git status --short
git -C ../cuda-metal-cli status --short
```

**Compiler source prerequisite:** the successful runs used local CuMetal fixes.
The historical base commit was `def27cb45d5a10f53fe4e473dc25dbb5f71c84b2`, plus
a source patch. A clean clone of that commit alone does not contain the fixes.
They are tracked upstream in issues
[#59](https://github.com/Lulzx/cuda-metal/issues/59),
[#60](https://github.com/Lulzx/cuda-metal/issues/60),
[#61](https://github.com/Lulzx/cuda-metal/issues/61),
[#64](https://github.com/Lulzx/cuda-metal/issues/64),
[#65](https://github.com/Lulzx/cuda-metal/issues/65),
[#66](https://github.com/Lulzx/cuda-metal/issues/66), and
[#67](https://github.com/Lulzx/cuda-metal/issues/67).
Use the existing patched worktree or a revision containing the reviewed fixes.
Do not assume the local branch name is published or that an arbitrary upstream
checkout reproduces these results.

For exact historical recovery, the local CuMetal evidence archive contains
`docs/experiments/llvm21-cli-validation/final-v10/compiler-source.patch`, SHA-256
`a3f9cc323f9a90f21c179aea62a1486829b3b598f8f84c869f833e55422d7d3b`.
It applies to the historical base, not on top of an already patched worktree.
The newer local parser compatibility adjustment is documented separately in
`docs/experiments/llvm21-pointer-inference.md`. If neither the patched checkout
nor the source archive is available, obtain the fixes before claiming an exact
reproduction; the historical binary hashes alone cannot rebuild missing source.

## 3. Set durable paths

Run the remaining commands from the vanity-miner-rs CuMetal worktree root.
Adjust the sibling CuMetal path if the worktree was named differently.

```sh
MINER_ROOT="$(pwd)"
CUMETAL_ROOT="$(cd ../cuda-metal-cli && pwd)"
CUMETAL_BUILD="$CUMETAL_ROOT/build-macos-guide"
ARTIFACT_ROOT="$MINER_ROOT/.cumetal-artifacts"
PTX_DIR="$ARTIFACT_ROOT/ptx-34803069395"
MODULE_DIR="$ARTIFACT_ROOT/modules-llvm21"
mkdir -p "$ARTIFACT_ROOT" "$MODULE_DIR"
```

`.cumetal-artifacts/` is ignored by Git. Keep downloads and generated binaries
there; keep commands, source revisions, hashes, and small reports in Git. Save
important external artifacts separately if you intend to delete the worktree.

## 4. Build CuMetal on the Mac

Prerequisites: Apple Silicon macOS, Apple Command Line Tools/SDK, Nix for the
checked-in CuMetal dev shell, CMake's dependencies supplied by that shell,
Python 3, and `gh` authenticated for artifact download. Use the Rust toolchain
specified by this repository when building the miner CLI.

Check the Apple tools first:

```sh
/usr/bin/xcrun --sdk macosx --show-sdk-path
/usr/bin/clang++ --version
```

The CuMetal `flake.nix` supplies CMake, Ninja, Python, LZ4, and Zstd. It uses Apple
Clang and the system SDK. It does not install Xcode or an Apple GPU driver.
This PTX-to-MSL flow was validated with standalone `metal`/`metallib` commands
absent: the runtime uses the Metal API to compile source. Other CuMetal paths,
including offline metallib generation and CUDA C++/NVVM import, have additional
requirements; a `cumetal doctor` failure about those tools is not this path's
numerical test.

Configure a new build directory, then build the two targets we need:

```sh
git -C "$CUMETAL_ROOT" submodule update --init --recursive

nix develop "$CUMETAL_ROOT" --command cmake \
  -S "$CUMETAL_ROOT" -B "$CUMETAL_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_OBJCXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_DISABLE_FIND_PACKAGE_LLVM=TRUE \
  -DCUMETAL_ENABLE_BINARY_SHIM=OFF

nix develop "$CUMETAL_ROOT" --command cmake \
  --build "$CUMETAL_BUILD" --target cumetal_runtime cumetalc -j 4
```

Disabling LLVM discovery here disables CuMetal's optional LLVM/NVVM importer,
not the typed PTX importer. LLVM 21 already did its job producing the PTX in CI.
The outputs are `$CUMETAL_BUILD/cumetalc` and
`$CUMETAL_BUILD/libcumetal.dylib`. The first Nix invocation can download substantial
cached dependencies even though it is not building Rust-CUDA.

Historically, `build-cli-apple` was configured similarly, with explicit LZ4/Zstd
paths from an older Nix store. Do not copy those machine-specific store paths as
a fresh-machine recipe. The clean configure above is intended to replace that
hidden CMake-cache dependency.

## 5. Download the exact PTX we validated

The original file came from the `cuda-compile` workflow in
[run 34803069395](https://github.com/brandonros/vanity-miner-rs/actions/runs/34803069395),
source commit `6a5b9e6b37390527b239baa8baae54d1445e920a`, branch
`poc/llvm21-portable-ptx`, artifact **`vanity-miner-aarch64-llvm21`**.
We re-downloaded it and verified it matches the original Downloads copy.

```sh
gh run download 34803069395 --repo brandonros/vanity-miner-rs \
  --name vanity-miner-aarch64-llvm21 --dir "$PTX_DIR"
PTX="$PTX_DIR/output-llvm21.ptx"
shasum -a 256 "$PTX"
python3 scripts/check-ptx-entries.py "$PTX"
```

Expected SHA-256:

```text
2840485fe193d39cfdbb2e5babd25f6772a08a2550843c20b3beba5a26185f4b
```

Size: 25,161,784 bytes. Inventory: **123 entrypoints = 118 numerical self-tests
+ one launch probe + four production mining kernels**. Artifact retention can
expire. If the old download is unavailable, use a preserved copy with that hash,
or build/download a new artifact and treat it as a new validation input.

For a fresh CI build, select a source ref intentionally rather than silently
substituting whatever branch happens to be newest:

```sh
gh workflow run cuda-compile.yaml --repo brandonros/vanity-miner-rs --ref CHOSEN_REF
gh run list --repo brandonros/vanity-miner-rs --workflow cuda-compile.yaml --branch CHOSEN_REF
# Inspect the intended run ID and source commit, then:
gh run watch RUN_ID --repo brandonros/vanity-miner-rs --exit-status
gh run download RUN_ID --repo brandonros/vanity-miner-rs \
  --name vanity-miner-aarch64-llvm21 --dir NEW_DOWNLOAD_DIRECTORY
```

The Linux LLVM 21 job runs the equivalent of:

```sh
nix develop .#v21 --command cargo build -p vanity-miner \
  --features "gpu,solana,bitcoin,ethereum,shallenge,self_test,llvm21" \
  --release --locked
```

It copies `target/llvm21/cuda-builder/nvptx64-nvidia-cuda/release/kernels.ptx`
to `output-llvm21.ptx`, then checks the entry inventory before upload. See the
[workflow](../.github/workflows/cuda-compile.yaml), [build script](../cli/build.rs),
and [entry check](../scripts/check-ptx-entries.py). New codegen-test or new-mode
branches may have a different inventory: record it and do not relabel those
artifacts as the tested 118-slot baseline.

## 6. One PTX file → one selected Metal module per command

For just Solana, translate once:

```sh
"$CUMETAL_BUILD/cumetalc" "$PTX" \
  --backend=cumetal-ir --ptx-strict \
  --entry kernel_find_solana_vanity_private_key --emit=msl \
  -o "$MODULE_DIR/kernel_find_solana_vanity_private_key.metal"
```

For all four mining modes, use the same PTX input four times, selecting a different
entry each time:

```sh
for entry in \
  kernel_find_better_shallenge_nonce \
  kernel_find_solana_vanity_private_key \
  kernel_find_ethereum_vanity_private_key \
  kernel_find_bitcoin_vanity_private_key
do
  "$CUMETAL_BUILD/cumetalc" "$PTX" \
    --backend=cumetal-ir --ptx-strict --overwrite \
    --entry "$entry" --emit=msl -o "$MODULE_DIR/$entry.metal"
done
```

Each invocation produces `ENTRY.metal` and `ENTRY.metal.cumetal-abi`. Keep them
together from the same successful compiler invocation. This step does not run
Rust-CUDA again and does not mutate or split the input PTX. CuMetal selects an
entry and lowers its supported code/dependencies; the selected name is not a
request to delete other kernels from the original artifact.

`--module-dir` consumes these outputs. Alternatively, the CLI accepts
`--ptx "$PTX" --cumetalc "$CUMETAL_BUILD/cumetalc"` and compiles/caches the
selected entry itself. For a timed experiment, precompiling makes it easier to
separate PTX translation from Metal compilation and execution.

## 7. Build the Mac CLI and run a correctness smoke test

```sh
cargo build --release -p vanity-miner --no-default-features \
  --features cumetal,self_test,shallenge,solana,ethereum,bitcoin --locked

CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0 CUMETAL_TRACE_GPU=1 \
./target/release/vanity-miner \
  --cumetal-library "$CUMETAL_BUILD/libcumetal.dylib" \
  --module-dir "$MODULE_DIR" \
  --batches 1 --threads-per-block 64 --blocks 16 --seed 1 --verify \
  solana-vanity BMz ''
```

Do not enable `gpu`/`llvm21` for this Mac CLI build: those select the NVIDIA
backend. The `cumetal` feature dynamically loads the CuMetal runtime.

This smoke test examines 1,024 candidates. Seed 1 / thread 0 has the known address
`BMzUNgLUwUUGch3cujgAd2Bn7cNsAZo8rNK4uQ9iK2sG`. Check for the Apple M5 GPU
provenance line and `verified=true guards=intact`, not merely exit status or a
successful compilation. `--verify` checks every candidate's match count against
CPU; reported matches also have their returned fields checked. The fixed seed
is a reproducible test fixture, not a recommendation for generating wallet keys.

## 8. Run the 60-second Solana benchmark

```sh
python3 scripts/benchmark-cumetal-solana.py \
  --cli target/release/vanity-miner \
  --library "$CUMETAL_BUILD/libcumetal.dylib" \
  --modules "$MODULE_DIR" \
  --threads 64 --blocks 16 --seconds 60 \
  --out "$ARTIFACT_ROOT/benchmark-60s-01"
```

Choose a new output directory for each attempt. Add `--tune` to test the short
layout sweep first. The script searches a rare prefix, excludes two warmup
batches, disables full CPU verification during timing, records guards and GPU
provenance, and stops after a completed batch reaches the requested interval.
It reports both completed candidates / wall time and candidates / Metal GPU
command-buffer time. It does not count first-time compilation as mining time.

On the validated 8-core M5, 64 x 16 was a good measured starting point; 32 x 32
was close. The 60-second result was 25,019 candidates/s end to end and 29,059
candidates/s over GPU timestamps. Larger launch dimensions were not automatically
faster. See [benchmark methodology, tuning table, and saved results](cumetal-benchmark.md).
For RTX 5090 comparisons, use the same kernel/workload, match conditions, and
warmup policy; compare the same timing metric and tune each GPU independently.

## 9. Full correctness validation and agent handoff

The original module's full compile inventory can be generated with:

```sh
python3 "$CUMETAL_ROOT/demos/rust-ptx/inventory.py" "$PTX" \
  --compiler "$CUMETAL_BUILD/cumetalc" --out "$ARTIFACT_ROOT/all-entries" \
  --jobs 4 --timeout 900
```

That inventories compilation only. Inspect `inventory.json` and require all
123 entries to produce both outputs; do not count a generated file as a GPU pass.
Then run the actual CLI suites:

```sh
python3 scripts/validate-cumetal-self-tests.py \
  --cli target/release/vanity-miner --library "$CUMETAL_BUILD/libcumetal.dylib" \
  --modules "$ARTIFACT_ROOT/all-entries" --out "$ARTIFACT_ROOT/self-tests-01"
python3 scripts/validate-cumetal-cli.py \
  --cli target/release/vanity-miner --library "$CUMETAL_BUILD/libcumetal.dylib" \
  --modules "$ARTIFACT_ROOT/all-entries" --out "$ARTIFACT_ROOT/mining-matrix-01"
```

The baseline passed all 118 numerical entries plus the probe in one CLI invocation,
and all four mining modes passed 36 scenarios / 69 CPU-verified batches. Those
are bounded correctness results, not exhaustive coverage of all future inputs.
The baseline full CLI invocation took about 34 minutes including Metal compilation.

For every future run, record source commit **and dirty diff**, PTX hash and entry
inventory, compiler/runtime/CLI binary hashes, commands and launch dimensions,
device/core count, warmup/timing method, and output reports. Revalidate after
changing compiler semantics, Rust kernels, artifact or launch layout. Do not carry
an old pass forward just because a filename or branch name stayed the same.

Existing compiler evidence is in the CuMetal worktree's
`docs/experiments/llvm21-cli-validation/final-v10/`. The downstream coverage
backlog is [issue #16](https://github.com/brandonros/vanity-miner-rs/issues/16).
Compiler extraction issues are linked above. Preserve the source fixes and
artifact provenance before cleaning up the old worktrees; deleting `/tmp`
outputs should only cost regeneration, not the knowledge needed to do it.

### Recipe verification (2026-09-14)

The clean `build-macos-guide` configure/build above succeeded. Re-downloading the
CI artifact reproduced the expected hash and all 123 entries. Solana translated
with that new compiler and ran with that new runtime on Apple M5: 1,024 candidates,
one expected match, `verified=true guards=intact`. See the
[smoke-test log](benchmarks/m5-solana/clean-build-smoke.log) and
[input hashes and scope](benchmarks/m5-solana/clean-build-smoke.json). This was a
fresh Solana smoke test, not a repeat of the full suite or 60-second benchmark.
