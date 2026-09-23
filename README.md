# vanity-miner-rs

Vanity address, key, and signature search in Rust. Backends: CPU and NVIDIA CUDA.

## Modes

| Cargo feature | Command |
| --- | --- |
| `solana` | `solana-vanity` |
| `bitcoin` | `bitcoin-vanity` |
| `ethereum` | `ethereum-vanity` |
| `shallenge` | `shallenge` |
| `p256-public-key` | `p256-public-key-vanity` |
| `p256-signature` | `p256-signature-vanity` |
| `rsa-pss` | `rsa-pss-signature-vanity` |

## Toolchain

`rust-toolchain.toml` pins a nightly with the `nvptx64-nvidia-cuda` target;
rustup installs it on first use. `nix develop` provides the same toolchain,
plus the CUDA headers and libclang that GPU builds need on Linux.

## CPU

```sh
cargo build -p vanity-miner --release --locked --no-default-features --features solana
./target/release/vanity-miner solana-vanity --prefix aaa
```

Combine mode features with commas. Run `<command> --help` for its options.
Address searches accept `--prefix` and `--suffix`; signature searches also need
`--key` and `--message`. Matches print to stdout; Ctrl-C stops the search.

## CUDA · Linux

Kernels and the runner build separately. PTX builds on any OS:

```sh
cargo ptx -p 'kernel-*'
```

This writes one `<module>.ptx` per crate in `crates/kernels/` to
`target/nvptx64-nvidia-cuda/release`; `-p kernel-solana` builds one.
`.cargo/config.toml` defines `cargo ptx` and the minimum GPU architecture (`sm_75`).

```sh
nix develop --command cargo build -p vanity-miner --release --locked --no-default-features --features gpu,solana
PTX_PATH=target/nvptx64-nvidia-cuda/release ./target/release/vanity-miner solana-vanity --prefix aaa
```

The runner loads `<module>.ptx` from `PTX_PATH` and needs an NVIDIA GPU and
driver. Without Nix, building it needs a CUDA toolkit (`CUDA_PATH` or
`/usr/local/cuda`) and libclang.

## Self-tests

```sh
cargo run -p vanity-miner --release --locked --no-default-features --features self_test -- self-test
```

`self_test` enables all 7 groups; `self_test_solana`, for example, enables one.
Use `self-test --list` to list checks and `self-test --check <name>` to select one.
Add `gpu` to the features and set `PTX_PATH` to run the same checks on CUDA.
