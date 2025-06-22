# vanity-miner-rs
GPU-accelerated vanity address generator for multiple blockchains

## How to use

```shell
cargo run --release -- solana-vanity aaa ""
cargo run --release -- ethereum-vanity 555555 "" # broken
cargo run --release -- bitcoin-vanity bc1qqqqq "" # broken
cargo run --release -- shallenge brandonros 000000000000cbaec87e070a04c2eb90644e16f37aab655ccdf683fdda5a6f96
```

## Apple Container

```shell
container system start
container build -t cuda-12-9-rust-builder
container run --rm -it --memory 8G -v $(pwd):/mnt cuda-12-9-rust-builder
cd /mnt
./scripts/build.sh
container system stop
```
