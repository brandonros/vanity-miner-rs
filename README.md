# vanity-miner-rs

Vanity address, key, and signature search in Rust on Apple GPUs. Kernels compile
with stock Rust through llvm-metal; Metal is the default application backend.
CPU references verify winners and support testing without a GPU.

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

## Build and run · macOS

Install Nix, then run from the repository root. The script uses the pinned
stable Rust and llvm-metal compiler, builds matching kernels and host artifacts,
and starts the search. `--mode` selects a mode from the table above.

```sh
./scripts/run-metal.sh --mode solana \
  --batches 2 --batch-size 33 --seed 583437459223573146 --verify \
  solana-vanity --prefix aaa --suffix NFC
./scripts/run-metal.sh --mode p256-public-key \
  --batches 2 --batch-size 1 --threads-per-group 1 --verify \
  p256-public-key-vanity --prefix ab
./scripts/run-metal.sh --mode rsa-pss \
  --batches 2 --batch-size 1 --threads-per-group 1 --verify \
  rsa-pss-signature-vanity --key private.pem --message message.bin
```

Omit `--batches` for continuous search. `--verify` compares every lane with the
CPU; winners are always CPU-verified. P-256 and RSA use OS cryptographic entropy
and reject `--seed`; start with small batches. Matches print to stdout; Ctrl-C
stops the search. Use `<command> --help` for options.

For compiler development, pass `--llvm-metal ../llvm-metal` before `--mode`.
To build a kernel separately:

```sh
nix develop --command python3 scripts/build-metal.py --mode shallenge
nix develop --command cargo build --locked --release -p vanity-miner
```

Kernel crates live in `crates/kernels/<mode>`. The stock NVPTX target supplies
LLVM bitcode to llvm-metal; no NVIDIA toolkit or driver is needed.

## CPU references and self-tests

```sh
nix develop --command cargo run -p vanity-miner --release --locked \
  --no-default-features --features self_test -- self-test
./scripts/run-metal.sh --mode self-test self-test
```

`self_test` enables all 8 groups; `self_test_solana`, for example, enables one.
For a Metal group, use `--mode self-test-solana`. Select named cases with
`self-test --check MODE.CHECK`; `self-test --list` lists the checks.

CPU-only builds work on Linux and macOS with `--no-default-features` and one or
more mode features, for example `--features solana`. Combine features with commas.
