# vanity-miner-rs
GPU-accelerated vanity address generator for multiple blockchains

## How to trigger CI

```shell
gh workflow run cuda-compile.yaml --ref cuda-oxide && sleep 5 && \
  gh run watch $(gh run list --workflow=cuda-compile.yaml --branch=cuda-oxide -L1 --json databaseId -q '.[0].databaseId') --exit-status
```

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
./target/release/vanity-miner self-test
./target/release/vanity-miner solana-vanity aaa ""
./target/release/vanity-miner ethereum-vanity 5555 ""
./target/release/vanity-miner bitcoin-vanity bc1qqqq ""
./target/release/vanity-miner shallenge brandonros 0000027f35458e484a48298988ceff6b7037418e4479ade56a08a13ac2823ebb
```

## Selective builds

The CUDA Oxide branch still enables all CLI modes by default. To build just
Shallenge on CPU, use `cargo build -p vanity-miner --no-default-features --features shallenge`.
For a selective GPU build, enable `gpu` plus the desired mode (`solana`, `bitcoin`,
`ethereum`, `shallenge`, or `self_test`) through the CUDA Oxide build toolchain.
Mode features forward to the matching kernel and logic features.

Shallenge supports usernames of 1–30 bytes; the nonce length adapts so
`username || '/' || nonce` occupies exactly 32 bytes. GPU contexts default to a
16 KiB stack; `STACK_SIZE` overrides it.

## Deployment scripts

- `scripts/vast-run.sh` retains the release download and compiler self-test workflow.
- `scripts/vast-run-local.sh` uploads a locally built binary and its PTX sidecar,
  patches the Nix ELF interpreter on the remote Linux host, then runs Shallenge.
  Override `VAST_HOST`, `VAST_PORT`, `LOCAL_BINARY`, or `LOCAL_PTX` as needed.
  Build with `scripts/build-gpu.sh` before using this script.
