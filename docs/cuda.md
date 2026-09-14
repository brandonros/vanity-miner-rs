# CUDA build and runtime guide

Run these commands from the repository root on `x86_64-linux` or
`aarch64-linux`. The repository provides Nix development shells; the previous
container instructions referenced a Dockerfile and build script that are not
present here.

## Build and validate

The shells read `rust-toolchain.toml` and provide the CUDA toolkit and the
selected LLVM toolchain. The host must supply the NVIDIA driver at runtime;
the toolkit's driver stubs are not a working runtime driver.

| Shell | CLI features for all modes | PTX target | Binary |
| --- | --- | --- | --- |
| `.#v7` | `gpu,solana,bitcoin,ethereum,shallenge,self_test` | `compute_89` | `target/llvm7/release/vanity-miner` |
| `.#v21` (also default) | `gpu,llvm21,solana,bitcoin,ethereum,shallenge,self_test` | `compute_100` | `target/llvm21/release/vanity-miner` |

```sh
nix develop .#v7 --command cargo build -p vanity-miner --features gpu,solana,bitcoin,ethereum,shallenge,self_test --release --locked
nix develop .#v7 --command ./target/llvm7/release/vanity-miner --help
nix develop .#v7 --command ./target/llvm7/release/vanity-miner self-test
```

For LLVM 21, use `.#v21`, add `llvm21` to the features, and run the binary under
`target/llvm21/`. Use a GPU and driver that support the emitted module. A
successful build or help command does not validate kernel execution. The
self-test checks isolated computations; it does not establish correctness of
concurrent production result collection.

The default mode feature is `shallenge`. To build an isolated mode, disable the
defaults and explicitly select both the runner and mode:

```sh
nix develop .#v7 --command cargo build -p vanity-miner --no-default-features --features gpu,bitcoin --release --locked
nix develop .#v7 --command ./target/llvm7/release/vanity-miner bitcoin-vanity bc1qqqq ""
```

The GPU runner is selected at compile time and has no CPU fallback. For CPU
builds and all mode examples, see [the README](../README.md).

## Module overrides

The default is the PTX embedded during the build. At runtime, `CUBIN_PATH` takes
precedence over `PTX_PATH`; if neither is set, embedded PTX is used. An explicit
file-loading failure is reported rather than falling back.

```sh
nix develop .#v7 --command env PTX_PATH=./output.ptx ./target/llvm7/release/vanity-miner bitcoin-vanity bc1qqqq ""
nix develop .#v7 --command env CUBIN_PATH=./output.cubin ./target/llvm7/release/vanity-miner bitcoin-vanity bc1qqqq ""
```

CUBIN files are loaded through the CUDA module-file API. Their GPU compatibility
must be established separately; selecting one does not convert it for a different
GPU. Both PTX and CUBIN overrides must implement the current kernel signatures
and the commands enabled in the host binary. Rebuild them when kernel interfaces
change. The embedded PTX build tracks `kernels/`, shared `logic/`, relevant
manifests and lockfiles, and `rust-toolchain.toml`.

## Runtime settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `BLOCKS_PER_SM` | `128` | Grid blocks per streaming multiprocessor |
| `THREADS_PER_BLOCK` | `256` | Threads per block |
| `STACK_SIZE` | `16384` | Requested per-thread stack limit in bytes |

These are current implementation defaults, not tuned performance guarantees.
The repository documents that composed k256 kernels need more than an 8-KiB
stack in its tested configuration. Use the default stack unless a change has
been validated for the selected mode and device. Launch settings currently need
additional bounds validation; keep values within the device and kernel limits.

Record the commit, mode features, LLVM version, GPU, driver, module identity,
launch settings, and validation result when comparing performance.
