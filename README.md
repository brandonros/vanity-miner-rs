# ed25519-vanity-rs
Ed25519 vanity generator with CUDA in Rust

## How to use

```shell
BLOCKS_PER_GRID="128"
THREADS_PER_BLOCK="128"
cargo run --release -- aa $BLOCKS_PER_GRID $THREADS_PER_BLOCK
```

