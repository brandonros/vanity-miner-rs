# vanity-miner-rs

Vanity address, key, and signature search in Rust. Backends: CPU, NVIDIA CUDA,
CuMetal, and direct Metal for Shallenge, Ethereum, Bitcoin, Solana and RSA modulus.

## Modes

| Cargo feature | Command |
| --- | --- |
| `solana` | `solana-vanity` |
| `bitcoin` | `bitcoin-vanity` |
| `ethereum` | `ethereum-vanity` |
| `shallenge` | `shallenge` |
| `p256-public-key` | `p256-public-key-vanity` |
| `p256-signature` | `p256-signature-vanity` |
| `rsa-modulus` | `rsa-modulus-vanity` |
| `rsa-pss` | `rsa-pss-signature-vanity` |

## CPU

Use the pinned Rust toolchain from the repository root:

```sh
cargo build -p vanity-miner --release --locked --no-default-features --features solana
./target/release/vanity-miner solana-vanity --prefix aaa
```

Combine mode features with commas; use `<command> --help` for options.
Address searches accept `--prefix` and `--suffix`; signature searches also need
`--key` and `--message`. Matches print to stdout; Ctrl-C stops the search.

## CUDA · Linux

```sh
nix develop .#v21 --command cargo build -p vanity-miner --release --locked --no-default-features --features llvm21,solana
./target/llvm21/release/vanity-miner solana-vanity --prefix aaa
```

For LLVM7, use `.#v7`, features `cuda-kernels,solana`, and `target/llvm7`.
Running requires a compatible NVIDIA GPU and driver.

Build the 8 production and 8 self-test PTX modules separately:

```sh
./scripts/build-ptx.sh 21
./scripts/build-ptx.sh 7
```

Bundles land in `artifacts/ptx-bundle-llvm<version>.tar.gz`. Use runner and PTX
artifacts from the same source revision. `PTX_PATH` selects an external bundle.

## Metal · macOS

```sh
./scripts/run-metal-shallenge.sh --batches 4 --batch-size 33 --seed 12345 --verify \
  shallenge --username brandonros \
  --target-hash ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
```

This builds matching device/host artifacts using `nix develop .#metal` (stable
Rust), then runs the `metal` backend. Omit `--batches` for continuous search.
`--verify` compares every lane with the CPU; winners are always CPU-verified.
The other Metal modes require a newer local llvm-metal checkout:

```sh
./scripts/run-metal.sh --llvm-metal ../llvm-metal --mode bitcoin \
  --batches 2 --batch-size 33 --seed 10088153575472065218 --verify \
  bitcoin-vanity --prefix bc1qg --suffix 6m
./scripts/run-metal.sh --llvm-metal ../llvm-metal --mode solana \
  --batches 2 --batch-size 33 --seed 583437459223573146 --verify \
  solana-vanity --prefix aaa --suffix NFC
./scripts/run-metal.sh --llvm-metal ../llvm-metal --mode rsa-modulus \
  --batches 2 --batch-size 1 --threads-per-group 1 --verify \
  rsa-modulus-vanity --prefix ab --steps-per-launch 1
```

RSA uses OS cryptographic entropy and rejects `--seed`; start with small batches.
Backend features `metal`, `gpu` and `cumetal` are mutually exclusive.

## CuMetal · macOS

```sh
nix develop .#cumetal --command cargo run -p vanity-miner --release --locked \
  --no-default-features --features cumetal,solana -- \
  --ptx /path/to/solana.ptx --batches 1 --verify solana-vanity --prefix aaa
```

The shell supplies the compiler/runtime pinned in `flake.lock`. `--ptx` accepts
a module or bundle directory. See the [issue matrix](docs/cumetal-issue-matrix.md)
for working workloads and remaining blockers. Do not combine `gpu` and `cumetal`
or use `--all-features`.

## Self-tests

```sh
cargo run -p vanity-miner --release --locked --no-default-features --features self_test -- self-test
```

`self_test` enables all 8 groups; `self_test_solana`, for example, enables one.
