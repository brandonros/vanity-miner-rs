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
nix develop .#v7 --command cargo build -p vanity-miner --release --locked --no-default-features --features cuda-kernels,solana,self_test
./target/llvm7/release/vanity-miner self-test
```

The default Nix shell is `v21`. `cuda-kernels` compiles and embeds PTX;
`llvm21` enables it automatically. `gpu` alone builds a runner without kernels:

```sh
nix develop .#runner --command cargo build -p vanity-miner --release --locked --features gpu,solana,self_test
PTX_PATH=/path/to/extracted/ptx ./target/runner/release/vanity-miner self-test
```

Each PTX module owns its build under `kernels/<module>/`: `build.rs` compiles the
nested `device/` crate, and `src/lib.rs` exposes the resulting PTX to the CLI.
Production and self-tests are separate sibling packages, such as
`kernels/bitcoin/` and `kernels/self-test-bitcoin/`.
Kernel packages can also be built independently, without compiling the CLI:

```sh
# Solana production PTX only, LLVM 21
nix develop .#v21 --command cargo build -p kernel-solana --release --locked --features llvm21

# Bitcoin production and self-test PTX, LLVM 7
nix develop .#v7 --command cargo build -p kernel-bitcoin -p kernel-self-test-bitcoin --release --locked

# Solana self-test PTX only, LLVM 21
nix develop .#v21 --command cargo build -p kernel-self-test-solana --release --locked --features llvm21
```

Each kernel package defaults to `cuda` and produces exactly one PTX module.
`kernel-<mode>` produces `<mode>.ptx`; `kernel-self-test-<mode>` produces
`self_test_<mode>.ptx` (PTX filenames use underscores).
The CLI disables that default and enables compilation
only with `cuda-kernels` or `llvm21`, so CPU and external-PTX runners do not
compile device code. Plain workspace commands default to `cli` and `logic`;
`--workspace` also selects the CUDA kernel packages.

Each kernel build watches its own device sources and shared inputs. Editing
another mode's device code does not invalidate it; editing shared `logic`,
compiler settings, or lockfiles can invalidate multiple modes. The build scripts
share settings through `kernels/build_support.rs`. Nested CUDA builds currently
share Rust-CUDA's target-directory lock, so separate packages do not guarantee
parallel GPU compilation.

The runner works with either LLVM bundle; select PTX for your GPU and use the
same source revision for runner and kernels.
For build profiling, set `NVVM_TIMING_DIR` to an absolute log directory and add
`--timings -vv` to the Cargo command. Rust-CUDA writes live phase logs and rustc
self-profiles there; Cargo saves HTML timing reports under each target directory's
`cargo-timings/`. Redirect the entire Nix command's output to capture setup and
dependency-build messages too. Profiling is off when the variable is unset.

Runtime overrides are checked in order: `CUBIN_PATH`, `PTX_PATH`, embedded PTX.
An invalid override fails rather than falling back; use artifacts matching the
binary's kernel interfaces and your GPU.

`gpu` selects CUDA without enabling search modes or a CPU fallback. All eight
modes support CUDA. CUDA searches retain their context, module, inputs,
and device buffers for the whole search. Devices run independently across matches;
a bounded result queue overlaps host verification with subsequent GPU batches.
The default launch size is four blocks per SM on the largest GPU.
All CUDA modes use `THREADS_PER_BLOCK` (default 256); partial batches round up
the block count, with excess lanes exiting immediately.
`BATCH_SIZE` overrides the candidate count per launch (1–1048576);
finite ranges use partial final batches. `vast-run.sh` forwards this setting.
The default stack is 64 KiB; `STACK_SIZE` overrides it. Multi-GPU throughput and
the new scheduling behavior still require hardware validation. CI builds one runner per Linux host architecture and one PTX bundle per LLVM version.

RSA modulus search uses the same `no_std` mining loop on CPU and GPU, with one
thin GPU kernel entry. Each CPU worker or GPU thread owns a p
factor and its constrained q range, and loops independently through preparation
and primality testing. Good factors and full-width range cursors stay on the GPU
between launches. No cross-thread work queue or block barriers are required.
Range construction still removes the interval forbidden by factor separation;
narrow patterns reject empty ranges before testing p's primality. Each thread
returns at most one pair per launch, erases that task, and starts fresh work on its
next launch. The host independently verifies completed pairs and exports keys.

`rsa-modulus-vanity --steps-per-launch 64` sets the per-thread work budget on both CPU and
GPU (default 64, range 1–1024). Each step either prepares one p candidate or tests one q. Increasing
the budget amortizes launches but delays results and cancellation; it does not
truncate unfinished ranges. `BATCH_SIZE` controls the number of persistent task
slots. Shared candidate-ID reservations keep device work disjoint across GPUs.
See [GPU search design](docs/gpu-search-pipeline.md) for validation and tradeoffs.

Rebuild the host binary and RSA PTX together. CUDA and CuMetal now require
`kernel_rsa_modulus_vanity` as the only entry in `rsa_modulus.ptx`; old four-stage
PTX overrides are incompatible. An existing GitHub Actions artifact does not
include local changes. Rebuild Metal sidecars and PTX/CUBIN overrides as well.

Address modes, P-256, and RSA-PSS retain their atomic count and single-result
protocol per launch. The RSA pipeline retains multiple factor results. Winning
lanes depend on GPU scheduling. All device-exported results are independently
verified on the host. Rebuild PTX/CUBIN overrides after kernel interface changes.

### Apple Silicon

To attempt PTX-to-Metal-source translation for all 16 modules in both built
bundles, with per-module timing and logs, run on macOS:

```sh
nix build .#cumetal --out-link .cumetal-artifacts/toolchain
python3 scripts/compile-ptx-cumetal.py
```

The script verifies the package revision and hashes against its manifest and
`flake.lock`, snapshots the bundle inputs, and records results under
`.cumetal-artifacts/ptx-compile-<id>/`. See `report.md`, `timings.tsv`, and
`metadata.json` there. Attempts run serially and continue after failures;
`--timeout SECONDS` changes the default 300-second limit per module. A nonzero
exit indicates at least one failure or timeout. This checks translation only;
it does not compile Metal binaries or execute GPU kernels.

The `cumetal` Nix shell builds the compiler and runtime together from the exact
fork revision in `flake.lock`, including its pinned submodule. Build the host
with `cumetal` instead of `gpu`, plus the desired mode features:

```bash
nix develop .#cumetal --command cargo run --release --locked -p vanity-miner \
  --no-default-features --features cumetal,shallenge -- \
  --ptx /absolute/path/to/ptx --batches 1 --verify \
  shallenge --username brandonros \
  --target-hash ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
```

All eight search commands and the numbered self-tests have CuMetal host dispatch.
`--ptx DIR` accepts separately compiled PTX modules; `--ptx FILE` accepts one
selected mode's module. Translation uses the pinned compiler's typed backend.
P-256 and RSA-PSS use structured candidate batches; RSA modulus uses one resumable
mining entry. Batch capacity is `--blocks` times `--threads-per-block`.
`--batches N` bounds kernel launches; `--verify` additionally checks candidates
against CPU logic, including RSA counters, results and saved task state. Wiring and compilation alone
do not establish GPU numerical correctness; validate the selected mode on your hardware.

The CLI embeds the locked CuMetal revision at build time and checks the package's
manifest and both artifact hashes before loading the runtime. It prints those
identities and the hash of each PTX snapshot it translates. Missing packages,
revision mismatches, and changed binaries fail before GPU initialization.
`nix develop` supplies the package path automatically. For a host built outside
that shell, build the same package with
`nix build .#cumetal --out-link .cumetal-artifacts/toolchain` and pass the absolute
`--cumetal-root "$PWD/.cumetal-artifacts/toolchain"` path.

The independent `--cumetalc`, `--cumetal-library`, and `--module-dir` options have
been removed so every run uses the verified compiler/runtime pair and translates
its PTX with that compiler. To advance the contribution revision, update the
CuMetal input ref if needed, run `nix flake update cumetal`, review the locked
commit, and rebuild the host in the shell. See [validation provenance rules](AGENTS.md).

Validation on 2026-09-15 of CuMetal `e5acf8cc0c65` on Apple M5 passed two
32-candidate batches each for Shallenge and P-256 public-key search with `--verify`.
Four production modes failed translation; RSA modulus and both RSA-PSS search
variants emitted Metal but timed out before completing a batch. All eight
self-test groups were attempted: 8 checks passed and 152 were blocked by
translation. The [16-row status](docs/cumetal-status.md) remains 3 true / 13 false,
but RSA-PSS production now gets past its earlier translation failure. See the
[complete report](docs/cumetal-validation.md) for fresh diagnostics, fix coverage,
validation limits, and exact artifact identities.
The [issue ownership matrix](docs/cumetal-issue-matrix.md) maps all 13 unresolved
workloads to upstream defects and distinguishes partial fixes from completed
validation. The full run above predates the later `92a9b8f4de23` and
`7d12f120a6b8` locks.

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
  CPU, CUDA, and CuMetal accept patterns across the full 256-byte modulus;
  overlapping prefix/suffix bytes must agree. Long-prefix experiments can leave one or zero
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
not the search commands. Names and indices are independent of feature selection.

For example, compile and run only RSA-PSS tests on the CPU:

```sh
cargo run -p vanity-miner --release --locked --no-default-features --features self_test_rsa_pss -- self-test
```

Add `gpu,llvm21` in the v21 shell for CUDA. Multiple self-test features still
produce separate PTX files. The same `self-test` command runs the selected groups
on the backend chosen at build time.

Select by stable name on any backend (repeat `--check` for multiple names):

```sh
./target/release/vanity-miner self-test --list
./target/release/vanity-miner self-test --check p256_public_key.order_minus_one
```

`--list` prints enabled names and descriptions without initializing a device.
Unknown or disabled names are errors. Numeric selectors and `--self-test-slot`
have been removed. CPU execution runs only selected checks; GPU execution runs
their containing mode kernels and reports the selected checks.

Define each check with `register_self_test!` in its mode's implementation files:

```rust
register_self_test! {
    /// p256 order minus one produces negative generator
    fn order_minus_one() -> u32 {
        let mut scalar = super::CRYPTO_FIXTURE_P256_ORDER;
        scalar[31] -= 1;
        u32::from(
            crate::crypto::p256::public_point(&black_box(scalar))
                == Some(super::scalar_fixtures::NEGATIVE_GENERATOR),
        )
    }
}
```

The macro applies `#[inline(never)]` and uses the doc comment as the description.
Keep inputs opaque with `black_box` inside the check. The function name supplies
the final component of its CLI selector; renaming it changes that selector.

`logic/src/self_test/registry.rs` lists each function path once, grouped by mode:

```rust
p256_public_key ("self_test_p256_public_key", "p256_public_key/mod.rs") {
    scalar_probes::order_minus_one;
}
```

This produces the name `p256_public_key.order_minus_one`. The explicit list owns
ordering and kernel membership; there is no automatic source-file discovery.
`registration.rs` generates the `Slot` enum, count, metadata, CPU dispatcher,
and direct-call runners. Each mode owns its implementations and fixtures;
shared RSA/P-256 constants live in `known_answers.rs`. Verify generated reference
fixtures with `python3 scripts/self-test-fixtures.py --check`.

Indices are assigned from the complete registry before feature selection, so
separately compiled mode kernels agree with the host. Rebuild both host and
self-test PTX after changing registry order or contents; older externally supplied
PTX is incompatible. Numeric references in historical validation reports and
regression comments describe the previous registry.

Eight `kernels/self-test-<mode>/device/src/lib.rs` kernels write their group's results.
Each mode is compiled into its own PTX file. Shared primitive checks have one
owner. The standalone `kernel-repro-nonce-sequence` package is outside the registry
and the normal PTX bundle; build it like any other kernel package.

The device entry points can also be tested on the host without compiling PTX:

```sh
cargo test --manifest-path kernels/self-test-solana/device/Cargo.toml --locked
```

All enabled checks run on CPU. `rsa_pss.end_to_end` is temporarily skipped on GPU;
its definition carries `#[gpu_skip = "reason"]`, which makes the macro omit its
device call. Its historical isolated LLVM 21 build took 429 seconds, peaked near
7.1 GiB, and produced about
30 MiB of PTX. These measurements predate the SHA-256 replacement. Isolated
SHA-256, PSS and CRT checks still run. An all-mode GPU run should pass every
check except this one documented skip. Remove `gpu_skip` once compilation is resolved.

Standalone GPU artifacts are written to `target/llvm21/release/ptx/` (or
`target/llvm7/release/ptx/`). The selected `self_test_<mode>.ptx` files hold the
self-tests. Production files are `solana.ptx`, `bitcoin.ptx`, `ethereum.ptx`,
`shallenge.ptx`, `p256_public_key.ptx`, `p256_signature.ptx`, `rsa_pss.ptx`, and
`rsa_modulus.ptx`, for the selected features. Each file is built separately;
RSA modulus contains four related entry points, while other production modules
contain one. There is no combined production PTX.
To override the embedded self-tests, set `PTX_PATH` to this directory and leave
`CUBIN_PATH` unset. CuMetal accepts the same directory through `--ptx`.

Two independent CI workflows run on pull requests, pushes to `main`/`master`,
and manual dispatches. `cuda-runners.yaml` builds two runners (x86_64 and aarch64)
without kernel compilation or embedded PTX. `cuda-kernels.yaml` builds two
CPU-independent PTX bundles (LLVM 7 and LLVM 21). Each workflow uploads its own
Actions artifacts; neither calls the other or publishes a release. Download a
runner and kernel bundle built from the same source revision, extract the bundle,
and set `PTX_PATH` to its directory at runtime.

Build the same full PTX bundles locally on Linux with Nix (no GPU required):

```sh
./scripts/build-ptx.sh 21 # default when no argument is given
./scripts/build-ptx.sh 7
```

The script reuses `target/llvm<version>` for incremental builds and writes
`artifacts/ptx-bundle-llvm<version>.tar.gz`. CI calls this same script.
Each invocation also creates `artifacts/ptx-timings-llvm<version>-<id>/`
with `build.log` and one TSV per completed kernel builder (module, wall seconds,
and `ok`/`failed`). Timings cover `CudaBuilder::build`, including device Cargo
work, but exclude Nix setup and host build dependencies. A fresh timing directory
reruns all 16 build scripts while retaining dependency caches; these are not
clean-build benchmarks. Interrupted builders may have no timing file. Cargo's
HTML timing report is saved under `target/llvm<version>/cargo-timings/`.
Set `CARGO_BUILD_JOBS=1` to build serially and reduce contention when measuring.
From macOS, with this checkout mounted writable in the `vanity-nixos` Lima VM,
run from the repository root:

```sh
limactl shell vanity-nixos -- bash "$PWD/scripts/build-ptx.sh" 21
```

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
removed. Feature selection is unchanged; self-tests use the named registry described above.
