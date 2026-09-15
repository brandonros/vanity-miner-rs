# vanity-miner-rs

Vanity address, key, and signature search in Rust, with CPU, NVIDIA CUDA, and
Apple Silicon CuMetal backends.

## Build

Run from the repository root using the pinned Rust toolchain.

```sh
# CPU: all search modes and self-tests
cargo build -p vanity-miner --release --locked --features solana,bitcoin,ethereum,shallenge,rsa-modulus,rsa-pss,p256-public-key,p256-signature,self_test
./target/release/vanity-miner --help
```

The default feature is `shallenge`. To build a single mode, use
`--no-default-features --features bitcoin`, for example. Builds share the same
output path unless you set `CARGO_TARGET_DIR`.

| Feature | Command | Search target |
| --- | --- | --- |
| `solana` | `solana-vanity` | Base58 address |
| `bitcoin` | `bitcoin-vanity` | Mainnet `bc1q` address |
| `ethereum` | `ethereum-vanity` | Hex address |
| `shallenge` | `shallenge` | Challenge hash |
| `rsa-modulus` | `rsa-modulus-vanity` | RSA-2048 modulus |
| `rsa-pss` | `rsa-pss-signature-vanity` | RSA-PSS signature bytes |
| `p256-public-key` | `p256-public-key-vanity` | P-256 public point |
| `p256-signature` | `p256-signature-vanity` | ECDSA signature bytes |

### NVIDIA CUDA

The Nix shells provide the Linux build toolchain for x86_64 and aarch64. Running
requires a compatible NVIDIA GPU and driver. Add mode features as needed:

```sh
# LLVM 21, compute_100 (Blackwell or later)
nix develop .#v21 --command cargo build -p vanity-miner --release --locked --no-default-features --features gpu,llvm21,solana,self_test
./target/llvm21/release/vanity-miner solana-vanity --prefix aaa

# LLVM 7, compute_89
nix develop .#v7 --command cargo build -p vanity-miner --release --locked --no-default-features --features gpu,solana,self_test
./target/llvm7/release/vanity-miner self-test
```

The default Nix shell is `v21`. PTX is compiled and embedded during the build.
For build profiling, set `NVVM_TIMING_DIR` to an absolute log directory and add
`--timings -vv` to the Cargo command. Rust-CUDA writes live phase logs and rustc
self-profiles there; Cargo saves HTML timing reports under each target directory's
`cargo-timings/`. Redirect the entire Nix command's output to capture setup and
dependency-build messages too. Profiling is off when the variable is unset.

Runtime overrides are checked in order: `CUBIN_PATH`, `PTX_PATH`, embedded PTX.
An invalid override fails rather than falling back; use artifacts matching the
binary's kernel interfaces and your GPU.

`gpu` selects CUDA without enabling search modes or a CPU fallback. All eight
modes support CUDA. RSA/P-256 CUDA searches retain their context, module, inputs,
and device buffers for the whole search. Devices run independently across matches;
a bounded result queue overlaps host verification with subsequent GPU batches.
The default launch size is four blocks per SM on the largest GPU.
All CUDA modes use `THREADS_PER_BLOCK` (default 256); partial batches round up
the block count, with excess lanes exiting immediately.
`CRYPTO_BATCH_SIZE` overrides the candidate count per launch (1–1048576);
finite ranges use partial final batches. `vast-run.sh` forwards this setting.
The default stack is 64 KiB; `STACK_SIZE` overrides it. Multi-GPU throughput and
the new scheduling behavior still require hardware validation. CI builds LLVM 7 and 21 for both
Linux host architectures.

RSA modulus search generates p candidates, constructs constrained q ranges,
tests factors, and advances persistent tasks on the GPU. Short ranges from many
factors share a launch; large ranges continue across launches without a 65536-q
cutoff. Range construction removes the interval forbidden by factor separation.
For narrow patterns it checks for an eligible range before testing p's primality.
Each factor task exports at most one pair and is then retired. The host supplies
search-wide bounds and entropy, independently verifies completed pairs, and exports
keys. Statistics show p candidates, probable p factors, nonempty ranges, and q
candidates separately, alongside verified matches. See [GPU search design](docs/gpu-search-pipeline.md).

Rebuild the host binary and RSA PTX together. CUDA now requires
`kernel_rsa_generate_v3`, `kernel_rsa_ranges_v3`, `kernel_rsa_search_v3`, and
`kernel_rsa_advance_v3` in `rsa_modulus.ptx`; older overrides fail symbol lookup.
The v2 entry remains for the CuMetal reference transport, which retains host range
preparation. An existing GitHub Actions artifact does not include local changes.

Address modes, P-256, and RSA-PSS retain their atomic count and single-result
protocol per launch. The RSA pipeline retains multiple factor results. Winning
lanes depend on GPU scheduling. All RSA/P-256 exported results are independently
verified on the host. Rebuild PTX/CUBIN overrides after kernel interface changes.

### Apple Silicon

Build with `cumetal` instead of `gpu`, plus the desired mode features. All eight
search commands and the numbered self-tests have CuMetal host dispatch. Supply
`--ptx DIR` containing separately compiled PTX modules, with `cumetalc` available,
or `--module-dir DIR` containing ENTRY.metal files and ABI sidecars. Translation
uses the typed backend. `--ptx FILE` also accepts one selected mode's module.
RSA/P-256 searches use 64-candidate shared-winner batches and the same host
verification as CUDA. `--batches N` bounds launches; `--verify` additionally checks
every candidate against CPU logic. Wiring and compilation alone do not establish
GPU numerical correctness; validate the selected mode on your hardware.
Current Apple M5 validation with CuMetal contribution commit `ddf496c` finds
compiler blockers in all four RSA/P-256 production modules: pointer/integer
mismatches in generated Metal for public-key/modulus searches, and undefined
registers at control-flow joins for signature searches. These commands have real
device dispatch but are not yet validated working searches on CuMetal.

`gpu` and `cumetal` are mutually exclusive; do not use `--all-features`.

## Search

```sh
./target/release/vanity-miner solana-vanity --prefix aaa
./target/release/vanity-miner bitcoin-vanity --prefix bc1qqqq
./target/release/vanity-miner ethereum-vanity --prefix 5555
./target/release/vanity-miner shallenge --username brandonros --target-hash 000000000000cbaec87e070a04c2eb90644e16f37aab655ccdf683fdda5a6f96
```

Address modes use `--prefix` and `--suffix`; omit either option to leave it
unconstrained. Ethereum requires even-length hex patterns without `0x`.
Shallenge uses `--username` and `--target-hash`. Positional search arguments are
no longer accepted.

### RSA and P-256

```sh
./target/release/vanity-miner rsa-modulus-vanity --prefix a --suffix b
./target/release/vanity-miner p256-public-key-vanity --prefix a --target xy

# Signature examples require existing rsa-private.pem / p256-private.pem inputs.
printf 'example:00000000' > message.bin
./target/release/vanity-miner rsa-pss-signature-vanity --key rsa-private.pem --message message.bin --prefix 0
./target/release/vanity-miner p256-signature-vanity --key p256-private.pem --message message.bin --nonce-offset 8 --nonce-length 8 --prefix a
```

These four modes accept case-insensitive hex prefixes and suffixes, including odd
lengths, without `0x`. Matches are verified, printed to the console, and the search
continues until Ctrl-C or exhaustion of a finite message/salt space.
`--threads N` selects CPU workers; Ctrl-C cancels between work batches.

| Mode | Pattern applies to |
| --- | --- |
| RSA modulus | 256-byte big-endian modulus, exactly 2048 bits, exponent 65537 |
| RSA-PSS | 256-byte signature, not its hash or salt |
| P-256 public key | `xy` (default), `x`, `y`, or `uncompressed` SEC1 bytes |
| P-256 signature | `raw` (default: 32-byte r then 32-byte s), `r`, or `s` |

- **RSA modulus:** constructs a constrained q progression instead of rejecting
  complete random keys. Winners receive primality, factor-distance, and key checks.
  Prefix and suffix can be combined (for example, `--prefix a3b6 --suffix abcd`).
  CPU and the CUDA pipeline accept patterns across the full 256-byte modulus;
  overlapping prefix/suffix bytes must agree. The legacy CuMetal transport requires
  suffixes shorter than 128 bytes. Long-prefix experiments can leave one or zero
  eligible q values per p and become much slower. Pattern width is not a prediction
  of feasibility or time to find a key. These constrained keys have no established
  security guarantee.
  The first hex digit must be `8`–`f` and the last must be odd. Validation rejects
  patterns whose interval and suffix cannot satisfy the required factor separation
  `|p - q| > 2^924`, including prefixes of 25 or more consecutive `f` digits.
  Passing validation does not guarantee suitable primes exist in the remaining space.
- **RSA-PSS:** uses SHA-256/MGF1-SHA-256 and searches salts by default.
  `--salt-length` defaults to 32, with a maximum of 222. Message-window search
  uses `--search-source message`; see command help for window and fixed-salt options.
- **P-256 signatures:** message-window search uses RFC 6979. `--s-form` selects
  `low` (default), `high`, or `either`; some consumers reject high-S signatures.
  `--search-source ephemeral` keeps the message fixed and searches secret nonces.
  Nonces are never exported: reusing an ECDSA nonce across messages can expose the key.

Every match is printed as a block of hex fields. `public_key` is the matched
P-256 point representation or RSA modulus, so the pattern is visible directly.
P-256 key matches include the private scalar; RSA key matches include the private
key as PKCS#8 DER hex. Signature matches include the signature and exact message
bytes; RSA-PSS also includes the salt. The miner writes no output files.

Signature searches take an existing PKCS#8 PEM key and message file through
`--key` and `--message`. Search counters keep advancing after each printed match.

CUDA signing puts private keys in device memory. RSA device operations are
unblinded and checked with the public exponent; host verification uses blinded
RSA. Secret buffers are cleared after synchronization, but complete device-memory
erasure is not guaranteed.

## Self-tests

```sh
cargo build -p vanity-miner --release --locked --no-default-features --features self_test
./target/release/vanity-miner self-test
```

`self_test` enables all eight self-test groups. Select individual groups with
`self_test_solana`, `self_test_bitcoin`, `self_test_ethereum`, `self_test_shallenge`,
`self_test_p256_public_key`, `self_test_p256_signature`, `self_test_rsa_pss`, or
`self_test_rsa_modulus`. These enable only their matching logic dependencies,
not the search commands. Selected checks retain their original slot numbers.

For example, compile and run only RSA-PSS tests on the CPU:

```sh
cargo run -p vanity-miner --release --locked --no-default-features --features self_test_rsa_pss -- self-test
```

Add `gpu,llvm21` in the v21 shell for CUDA. Multiple self-test features still
produce separate PTX files. The same `self-test` command runs the selected groups
on the backend chosen at build time.

The `logic/src/self_test/` folder has one file per mode: `solana.rs`, `bitcoin.rs`,
`ethereum.rs`, `shallenge.rs`, `p256_public_key.rs`, `p256_signature.rs`,
`rsa_pss.rs`, and `rsa_modulus.rs`. Each contains that mode's primitive, pipeline,
boundary, and regression checks. Shared RSA/P-256 constants live in `fixtures.rs`.
`mod.rs` owns all 160 slot labels and the CPU dispatcher. Eight
`kernels/src/self_test_<mode>.rs` kernels each run once and write their checks to
the same numbered slots. Each mode is compiled separately into its own PTX file;
the CLI embeds the selected modules and loads the appropriate one for each group.
Each kernel feature enables only its matching logic self-tests and mode dependencies.
The standalone nonce reproduction uses the kernel-only `repro_nonce_sequence`
feature; it is not included in mode self-tests.
Shared primitives and compiler regressions have one
owner; checks are not duplicated across modes. With all groups enabled, CPU runs report 160 passes with no skips.

GPU slot 155 (RSA-PSS end-to-end candidate pipeline) is temporarily disabled and
reported as SKIP. Its isolated LLVM 21 build took 429 seconds, peaked near 7.1 GiB,
and produced about 30 MiB of PTX; the combined RSA-PSS self-test could run out of
memory. These measurements predate the SHA-256 replacement. Slots 135–144 still
exercise SHA-256, PSS and CRT on GPU; the full slot 155 fixture still runs on CPU.
With all other checks passing, GPU runs report 156 passes and one skip. Restore
its kernel call once the compile-time blow-up is resolved.

Standalone GPU artifacts are written to `target/llvm21/release/ptx/` (or
`target/llvm7/release/ptx/`). The selected `self_test_<mode>.ptx` files hold the
self-tests. Production files are `solana.ptx`, `bitcoin.ptx`, `ethereum.ptx`,
`shallenge.ptx`, `p256_public_key.ptx`, `p256_signature.ptx`, `rsa_pss.ptx`, and
`rsa_modulus.ptx`, for the selected features. Each file is built separately with
one feature and contains one kernel entry. There is no combined production PTX.
To override the embedded self-tests, set `PTX_PATH` to this directory and leave
`CUBIN_PATH` unset. CuMetal accepts the same directory through `--ptx`.

| Slots | Coverage |
| --- | --- |
| 0–117 | Original primitives, pipelines, and compiler regressions |
| 118–125 | P-256 key derivation, points, encoding, invalid scalars |
| 126–134 | P-256 signing, RFC 6979, nonce validity, S forms, message carry |
| 135–144 | SHA-256, MGF1, PSS salt boundaries, CRT and rejection |
| 145–152 | RSA multiplication, progression, primality, boundaries |
| 153–156 | Full RSA/P-256 candidate pipelines against fixed CPU references |
| 157–159 | RSA device range construction, cursor retirement, and factor derivation |

These kernels test candidate logic, not production CUDA argument passing or batch
buffer layouts. CPU passes and successful CUDA compilation do not establish GPU
numerical correctness.

## Source layout

`logic/src` contains shared CPU/GPU code:

- `modes/`: one search implementation per mode, matching the kernel files.
- `crypto/`: hashes, elliptic curves, and RSA arithmetic.
- `encoding/`: base58 and bech32.
- `search/`: patterns, candidate results, derivation, and RNG helpers.
- `self_test/`: primitive, pipeline, compiler, P-256, and RSA checks, plus fixtures.

Imports follow the folders, for example `logic::modes::p256_public_key_vanity`
and `logic::search::hex_pattern::HexPattern`. Root compatibility exports are
removed. Feature selection and self-test slot numbering are unchanged.
