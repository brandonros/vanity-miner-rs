# vanity-miner-rs
GPU-accelerated vanity address generator for multiple blockchains.

Run commands from the repository root. Modes are selected at build time:
the default build includes only `shallenge`. The `gpu` feature selects the NVIDIA
CUDA runner; it does not enable additional modes and has no CPU fallback.
The optional `cumetal` backend runs prebuilt Rust-CUDA PTX on Apple Silicon;
see [the CuMetal guide](docs/cumetal.md). Select only one GPU backend.

## CPU mode (no CUDA required)

Use the Rust toolchain specified in `rust-toolchain.toml`. These commands assume
`CARGO_TARGET_DIR` is unset; otherwise use the binary in that target directory.

```sh
# Build all search modes and the self-test command.
cargo build -p vanity-miner --features solana,bitcoin,ethereum,shallenge,self_test --release --locked

./target/release/vanity-miner solana-vanity aaa ""
./target/release/vanity-miner ethereum-vanity 5555 ""
./target/release/vanity-miner bitcoin-vanity bc1qqqq ""
./target/release/vanity-miner shallenge brandonros 000000000000cbaec87e070a04c2eb90644e16f37aab655ccdf683fdda5a6f96
./target/release/vanity-miner self-test
./target/release/vanity-miner --help
```

For a single mode, use `--no-default-features --features solana`, `bitcoin`,
`ethereum`, or `shallenge`. For example:

```sh
cargo build -p vanity-miner --no-default-features --features bitcoin --release --locked
./target/release/vanity-miner bitcoin-vanity bc1qqqq ""
```

Each build replaces the binary at the same target path. `--features self_test`
adds the self-test command; it does not expose the other search commands unless
their CLI features are enabled too. Ethereum patterns currently require an even
number of hex digits, without `0x`. Bitcoin examples use the supported lowercase
mainnet `bc1q` prefix and an empty suffix.

## GPU mode (Linux with CUDA)

The Nix development shells support `x86_64-linux` and `aarch64-linux` and provide
the build toolchain. Running requires a compatible NVIDIA GPU and host driver.
See [the CUDA guide](docs/cuda.md) for runtime settings and validation.

```sh
# LLVM 7, compute_89; default shell is also available as .#v7.
nix develop .#v7 --command cargo build -p vanity-miner --features gpu,solana,bitcoin,ethereum,shallenge,self_test --release --locked
nix develop .#v7 --command ./target/llvm7/release/vanity-miner self-test
nix develop .#v7 --command ./target/llvm7/release/vanity-miner solana-vanity aaa ""

# LLVM 21, compute_100.
nix develop .#v21 --command cargo build -p vanity-miner --features gpu,llvm21,solana,bitcoin,ethereum,shallenge,self_test --release --locked
nix develop .#v21 --command ./target/llvm21/release/vanity-miner self-test
```

Shell builds use `target/llvm7/` and `target/llvm21/` respectively. `--all-features`
also enables LLVM 21 and therefore requires the v21 shell. To build only Bitcoin
with LLVM 7, use `--no-default-features --features gpu,bitcoin` in the v7 shell;
for LLVM 21, use `--no-default-features --features gpu,llvm21,bitcoin` in v21.

PTX is compiled and embedded automatically. An override is optional:

```sh
nix develop .#v7 --command env PTX_PATH=./output.ptx ./target/llvm7/release/vanity-miner solana-vanity aaa ""
nix develop .#v7 --command env CUBIN_PATH=./output.cubin ./target/llvm7/release/vanity-miner solana-vanity aaa ""
```

Module selection is **CUBIN_PATH, then PTX_PATH, then embedded PTX**. A selected
file that cannot be loaded produces an error; it does not fall back. Override
files must match the host binary's kernel signatures and enabled modes, and the
GPU/driver must support the selected module. Rebuild overrides after changing
kernel interfaces.

CI builds both LLVM backends for both Linux host architectures. Release assets
use explicit `-llvm7` or `-llvm21` suffixes: two host binaries and one standalone
PTX file per LLVM version. The host architecture suffix does not identify the GPU.

This branch pins Rust-CUDA to `d2104a0a49252068292985e5e63328f522415c4b` from
`poc/portable-ptx-export`, based on the LLVM 21.1.8 / CUDA 13.3 upgrade.
The default Nix shell is `v21`; LLVM 19 has been replaced by the `llvm21` feature.
Modern merged-module DCE is enabled by the backend by default. Optional cleanup
and inlining remain disabled. LLVM 7 builds remain available through `.#v7`.
