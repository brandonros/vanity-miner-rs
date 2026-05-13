# vanity-miner-rs
GPU-accelerated vanity address generator for multiple blockchains

## How to build

```shell
cargo install --path ~/cuda-oxide/crates/cargo-oxide --force
cargo oxide build --features gpu --arch sm_89
```

## How to use

### CPU mode (no CUDA required)
```shell
# Build CPU-only binary
cargo build -p vanity-miner --release

# Run
./target/release/vanity-miner solana-vanity aaa ""
./target/release/vanity-miner ethereum-vanity 5555 ""
./target/release/vanity-miner bitcoin-vanity bc1qqqq ""
./target/release/vanity-miner shallenge brandonros 0000027f35458e484a48298988ceff6b7037418e4479ade56a08a13ac2823ebb
```
