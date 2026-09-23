# vanity-miner-rs

Vanity address, key, and signature search in Rust. Backends: CPU, and NVIDIA CUDA.

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

Combine mode features with commas. Run `<command> --help` for its options.
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

## Self-tests

```sh
cargo run -p vanity-miner --release --locked --no-default-features --features self_test -- self-test
```

`self_test` enables all 8 groups; `self_test_solana`, for example, enables one.
Use `self-test --list` to list checks and `self-test --check <name>` to select one.
GPU builds use the same command with their backend feature and matching PTX.
