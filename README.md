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
# Build GPU-enabled binary
cargo build -p vanity-miner --features gpu --release

# Run (requires CUBIN_PATH or PTX_PATH environment variable)
CUBIN_PATH=./output.cubin ./target/release/vanity-miner solana-vanity aaa ""
```

### CLI Help
```shell
./target/release/vanity-miner --help
```
