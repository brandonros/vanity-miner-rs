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
For Shallenge, Ethereum, Bitcoin, and Solana, the script defaults to 65,536
candidates per batch and 64 threads per group. Override these with `--batch-size`
and `--threads-per-group`.

```sh
just run solana \
  --batches 2 --batch-size 33 --seed 583437459223573146 --verify \
  solana-vanity --prefix aaa --suffix NFC
just run p256-public-key \
  --batches 2 --batch-size 1 --threads-per-group 1 --verify \
  p256-public-key-vanity --prefix ab
just run rsa-pss \
  --batches 2 --batch-size 1 --threads-per-group 1 --verify \
  rsa-pss-signature-vanity --key private.pem --message message.bin
```

It creates temporary signing keys and fails on errors, missing output, or timeout.
RSA modulus still needs to find a prime pair.

Omit `--batches` for continuous search. `--verify` compares every lane with the
CPU; winners are always CPU-verified. P-256 and RSA use OS cryptographic entropy
and reject `--seed`; start with small batches. Matches print to stdout; Ctrl-C
stops the search. Use `--exit-on-first-match` to stop after one verified match.
Use `<command> --help` for options.

Selective inlining is the default; `just build <mode> --inlining all` opts
into full inlining. For compiler development, enter the shell with
`nix develop --override-input llvm-metal path:../llvm-metal`. To build a kernel separately:

```sh
nix develop --command just build shallenge
nix develop --command cargo build --locked --release -p vanity-miner
```

Kernel crates live in `crates/kernels/<mode>`. The stock NVPTX target supplies
LLVM bitcode to llvm-metal; no NVIDIA toolkit or driver is needed.

## CPU references and self-tests

```sh
nix develop --command cargo run -p vanity-miner --release --locked \
  --no-default-features --features self_test -- self-test
just test self_test::
```

`self_test` enables all 8 groups; `self_test_solana`, for example, enables one.
`just test` builds every bundle and runs the ignored GPU tests with
`cargo test`; `--skip-build` reuses bundles, and a test name filter selects tests.
For one group, use `just run self-test-solana self-test`.
Select named cases with
`self-test --check MODE.CHECK`; `self-test --list` lists the checks.

CPU-only builds work on Linux and macOS with `--no-default-features` and one or
more mode features, for example `--features solana`. Combine features with commas.
CPU searches accept a common `--threads N`; Metal builds select `metal` plus the
mode feature.
