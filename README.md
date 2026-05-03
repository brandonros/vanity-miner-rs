# vanity-miner-rs
GPU-accelerated vanity address generator for multiple blockchains

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
