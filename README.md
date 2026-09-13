# vanity-miner-rs
GPU-accelerated vanity address generator for multiple blockchains

## How to build

```shell
# Default: LLVM 7, compute_89
nix develop --command cargo build -p vanity-miner --features gpu --release

# Opt-in: LLVM 19, compute_100 (Blackwell or later)
nix develop .#v19 --command cargo build -p vanity-miner --features gpu,llvm19 --release
```

The default shell is also available as `.#v7`. Shell builds use
`target/llvm7/` and `target/llvm19/` respectively. Add
`solana,bitcoin,ethereum,shallenge,self_test` to build every kernel;
`--all-features` also enables LLVM 19 and therefore requires the v19 shell.
CI builds both backends for both Linux host architectures. Artifacts and release assets
use explicit `-llvm7` or `-llvm19` suffixes: two host binaries and one standalone
PTX file per LLVM version.

## How to use

### CPU mode (no CUDA required)
```shell
# Build CPU-only binary
cargo build -p vanity-miner --release

# Run
./target/release/vanity-miner solana-vanity aaa ""
./target/release/vanity-miner ethereum-vanity 5555 ""
./target/release/vanity-miner bitcoin-vanity bc1qqqq ""
./target/release/vanity-miner shallenge brandonros 000000000000cbaec87e070a04c2eb90644e16f37aab655ccdf683fdda5a6f96
```

### GPU mode (requires CUDA)
```shell
# Build GPU-enabled binary (PTX is built and embedded automatically)
cargo build -p vanity-miner --features gpu --release

# Run — the binary is self-contained, no env vars needed
./target/release/vanity-miner solana-vanity aaa ""

# Optional: override the embedded PTX with a hand-built one
PTX_PATH=./output.ptx ./target/release/vanity-miner solana-vanity aaa ""
CUBIN_PATH=./output.cubin ./target/release/vanity-miner solana-vanity aaa ""
```

### CLI Help
```shell
./target/release/vanity-miner --help
```
