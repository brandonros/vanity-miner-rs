# vanity-miner-rs
GPU-accelerated vanity address generator for multiple blockchains.

## Cryptographic vanity modes

Four new commands have CPU runners and CUDA kernels/host dispatch. Shared logic
and CPU integration are tested. CI compiles all four new CUDA modes on both
LLVM backends; hardware validation and GPU performance measurements remain pending. See [implementation status](docs/new-modes-plan.md).

```sh
cargo build -p vanity-miner --no-default-features --features rsa-modulus,rsa-pss,p256-public-key,p256-signature --release --locked

./target/release/vanity-miner rsa-modulus-vanity --prefix a --suffix b --private-out rsa-private.pem --public-out rsa-public.pem
./target/release/vanity-miner p256-public-key-vanity --prefix a --target xy --private-out p256-private.pem --public-out p256-public.pem --public-format spki-pem

printf 'example:00000000' > message.bin
./target/release/vanity-miner rsa-pss-signature-vanity --key rsa-private.pem --message message.bin --prefix 0 --signature-out rsa-signature.bin --salt-out rsa-salt.bin
./target/release/vanity-miner p256-signature-vanity --key p256-private.pem --message message.bin --nonce-offset 8 --nonce-length 8 --prefix a --signature-out p256-signature.bin --message-out winning-message.bin --der-out p256-signature.der
```

Each mode has its own feature: `rsa-modulus`, `rsa-pss`, `p256-public-key`, or
`p256-signature`. New searches stop after the first independently verified winner.
`--threads N` selects CPU workers; Ctrl-C cancels the search and joins workers.
Statistics name keys, q candidates, salts, messages, or ephemeral nonces tested.
Elapsed rates include setup and winner verification, so short searches are not
steady-state benchmarks.

Patterns use case-insensitive hexadecimal without `0x`. `--prefix` and `--suffix`
both accept odd digit counts. Contradictory overlaps and excessive lengths are
rejected. Matching applies to these exact byte strings:

| Command | Matched bytes |
| --- | --- |
| `rsa-modulus-vanity` | Unsigned 256-byte big-endian modulus n; exactly 2048 bits, odd, e=65537 |
| `rsa-pss-signature-vanity` | Raw 256-byte big-endian RSA signature, not its hash or salt |
| `p256-public-key-vanity` | `xy` (default): X followed by Y, 64 bytes; `x`/`y`: 32 bytes; `uncompressed`: 04 followed by X and Y, 65 bytes |
| `p256-signature-vanity` | `raw` (default): 32-byte r followed by 32-byte s; `r`/`s`: the selected 32-byte component |

Generic brute-force work is approximately `16^d` for d independent matched hex
digits: four digits mean roughly 65,536 candidates; eight mean roughly 4.3 billion.
Overlapping bits count once; fixed format bits do not add work. RSA modulus
construction is different: it chooses a random feasible p and restricts q to the
prefix interval and suffix residue class, so its q candidates already satisfy
the pattern. It does not repeatedly generate and reject complete random keys.
Constraints leaving insufficient room for 256 bits of q candidate entropy are
rejected. Every winning key receives additional OS-random primality checks,
factor-distance checks, component validation, and a blinded sign/verify check.

RSA-PSS supports SHA-256 and MGF1-SHA-256. `--salt-length` defaults to 32 (maximum
222 for RSA-2048). The salt search enumerates distinct salts from a random starting
value; a zero-byte salt has one candidate. For a message-window search, select
`--search-source message --nonce-offset N --nonce-length N --message-out FILE`.
Supply `--fixed-salt-hex HEX` of the requested salt length, or let the search
generate one fixed salt. `--salt-out` saves the exact winning salt in both cases.
The CPU private operation is blinded; the explicit salt remains fully controlled. GPU
signing places private-key material in device memory. The device CRT operation is
unblinded and checks each result with the public exponent; host reconstruction
uses blinded RSA and independent PSS verification. Secret transport buffers are
cleared after synchronization, but device faults and compiler-generated copies
prevent a guarantee of complete erasure.

P-256 signature message-window search uses deterministic RFC 6979 signing and is
the recommended interface. `--search-source ephemeral` keeps the message fixed
and derives secret nonces from fresh OS entropy, bound to the key, message,
worker, and counter. **Reusing an ECDSA ephemeral nonce across different messages
can reveal the private key.** An ephemeral nonce is different from the public
message window: it is never printed or exported. Search seeds are never saved.
Both private scalars and ephemeral nonces use rejection sampling without modulo
bias. `--s-form low` is the default; `high` selects n-s when necessary, and `either`
tests both representations and emits the one that matched. Some consumers reject
high-S signatures. Optional DER output encodes the emitted signature; pattern
semantics always apply to the raw fixed-width bytes.

Private keys use PKCS#8 PEM; RSA public keys use SPKI PEM. P-256 public output is
SEC1 uncompressed bytes by default, or SPKI PEM with `--public-format spki-pem`.
Private files are staged with Unix mode 0600 and published atomically; existing
outputs require `--force`. Inputs cannot be replaced by output paths. Each file
is atomic individually; companion output can remain if a later publication fails.
All files are staged before publication, and private keys or raw signatures are
published last. No private key, factor, scalar, search seed, or ephemeral nonce is
printed in statistics or normal error messages.

For CUDA, add `gpu` to the same mode features using the pinned Linux/CUDA
toolchain. New searches use 64-candidate batches, rotate across available devices,
and observe Ctrl-C between synchronized batches. GPU performance has not been
measured; the default 64-KiB stack limit can be overridden with `STACK_SIZE`.
Add `self_test` and run `self-test` to exercise crypto differential fixtures
through the selected backend before searches.

Independent verification examples:

```sh
openssl rsa -in rsa-private.pem -check -noout
openssl dgst -sha256 -verify rsa-public.pem -signature rsa-signature.bin -sigopt rsa_padding_mode:pss -sigopt rsa_pss_saltlen:32 message.bin
openssl dgst -sha256 -verify p256-public.pem -signature p256-signature.der winning-message.bin
```

Run commands from the repository root. Modes are selected at build time:
the default build includes only `shallenge`. The `gpu` feature selects the NVIDIA
CUDA runner; it does not enable additional modes and has no CPU fallback.
The optional `cumetal` backend runs prebuilt Rust-CUDA PTX on Apple Silicon;
see [the CuMetal guide](docs/cumetal.md). For the complete build, artifact download,
translation, validation, and benchmark workflow, use the
[CuMetal reproduction runbook](docs/cumetal-reproduction.md). Select only one GPU backend.

## CPU mode (no CUDA required)

Use the Rust toolchain specified in `rust-toolchain.toml`. These commands assume
`CARGO_TARGET_DIR` is unset; otherwise use the binary in that target directory.

```sh
# Build all search modes and the self-test command.
cargo build -p vanity-miner --features solana,bitcoin,ethereum,shallenge,self_test,rsa-modulus,rsa-pss,p256-public-key,p256-signature --release --locked

./target/release/vanity-miner solana-vanity aaa ""
./target/release/vanity-miner ethereum-vanity 5555 ""
./target/release/vanity-miner bitcoin-vanity bc1qqqq ""
./target/release/vanity-miner shallenge brandonros 000000000000cbaec87e070a04c2eb90644e16f37aab655ccdf683fdda5a6f96
./target/release/vanity-miner self-test
./target/release/vanity-miner --help
```

For a single mode, use `--no-default-features --features solana`, `bitcoin`,
`ethereum`, or `shallenge`. For example:

```sh
cargo build -p vanity-miner --no-default-features --features bitcoin --release --locked
./target/release/vanity-miner bitcoin-vanity bc1qqqq ""
```

Each build replaces the binary at the same target path. `--features self_test`
adds the self-test command; it does not expose the other search commands unless
their CLI features are enabled too. Ethereum patterns currently require an even
number of hex digits, without `0x`. Bitcoin examples use the supported lowercase
mainnet `bc1q` prefix and an empty suffix.

## GPU mode (Linux with CUDA)

The Nix development shells support `x86_64-linux` and `aarch64-linux` and provide
the build toolchain. Running requires a compatible NVIDIA GPU and host driver.
See [the CUDA guide](docs/cuda.md) for runtime settings and validation.

```sh
# LLVM 7, compute_89; select the legacy shell explicitly.
nix develop .#v7 --command cargo build -p vanity-miner --features gpu,solana,bitcoin,ethereum,shallenge,self_test,rsa-modulus,rsa-pss,p256-public-key,p256-signature --release --locked
nix develop .#v7 --command ./target/llvm7/release/vanity-miner self-test
nix develop .#v7 --command ./target/llvm7/release/vanity-miner solana-vanity aaa ""

# LLVM 21, compute_100.
nix develop .#v21 --command cargo build -p vanity-miner --features gpu,llvm21,solana,bitcoin,ethereum,shallenge,self_test,rsa-modulus,rsa-pss,p256-public-key,p256-signature --release --locked
nix develop .#v21 --command ./target/llvm21/release/vanity-miner self-test
```

Shell builds use `target/llvm7/` and `target/llvm21/` respectively. `--all-features`
enables mutually exclusive CUDA and CuMetal backends; select features explicitly. To build only Bitcoin
with LLVM 7, use `--no-default-features --features gpu,bitcoin` in the v7 shell;
for LLVM 21, use `--no-default-features --features gpu,llvm21,bitcoin` in v21.

PTX is compiled and embedded automatically. An override is optional:

```sh
nix develop .#v7 --command env PTX_PATH=./output.ptx ./target/llvm7/release/vanity-miner solana-vanity aaa ""
nix develop .#v7 --command env CUBIN_PATH=./output.cubin ./target/llvm7/release/vanity-miner solana-vanity aaa ""
```

Module selection is **CUBIN_PATH, then PTX_PATH, then embedded PTX**. A selected
file that cannot be loaded produces an error; it does not fall back. Override
files must match the host binary's kernel signatures and enabled modes, and the
GPU/driver must support the selected module. Rebuild overrides after changing
kernel interfaces.

CI builds both LLVM backends for both Linux host architectures. Release assets
use explicit `-llvm7` or `-llvm21` suffixes: two host binaries and one standalone
PTX file per LLVM version. The host architecture suffix does not identify the GPU.

This branch pins Rust-CUDA to `d2104a0a49252068292985e5e63328f522415c4b` from
`poc/portable-ptx-export`, based on the LLVM 21.1.8 / CUDA 13.3 upgrade.
The default Nix shell is `v21`; LLVM 19 has been replaced by the `llvm21` feature.
Modern merged-module DCE is enabled by the backend by default. Optional cleanup
and inlining remain disabled. LLVM 7 builds remain available through `.#v7`.

The RSA and P-256 commands support CPU and NVIDIA CUDA builds. CuMetal
currently exposes only Solana, Bitcoin, Ethereum, Shallenge, and its existing
self-tests; enabling RSA/P-256 features does not add those commands to CuMetal.

## Focused Rust GPU reproductions

The `self_test` build also exports a small runtime-input kernel for the observed
nonce-generation failure. It preserves the existing 118
known-answer slots and can be run independently with raw mismatch reporting.
See the [coverage audit and reproduction guide](docs/codegen-repros.md).
