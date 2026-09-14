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
./target/llvm21/release/vanity-miner solana-vanity aaa ""

# LLVM 7, compute_89
nix develop .#v7 --command cargo build -p vanity-miner --release --locked --no-default-features --features gpu,solana,self_test
./target/llvm7/release/vanity-miner self-test
```

The default Nix shell is `v21`. PTX is compiled and embedded during the build.
Runtime overrides are checked in order: `CUBIN_PATH`, `PTX_PATH`, embedded PTX.
An invalid override fails rather than falling back; use artifacts matching the
binary's kernel interfaces and your GPU.

`gpu` selects CUDA without enabling search modes or a CPU fallback. All eight
modes support CUDA. RSA/P-256 searches use 64-candidate batches per device and a
64-KiB default stack; `STACK_SIZE` overrides the stack size. Their GPU performance
and numerical correctness remain unverified. CI builds LLVM 7 and 21 for both
Linux host architectures.

All eight modes count matches atomically and return one winner per launch.
The winning lane depends on GPU scheduling. RSA/P-256 winners are independently
verified on the host; an RSA candidate rejected by stronger primality checks is
discarded with the rest of its batch, and the search advances to the next batch.
Rebuild PTX/CUBIN overrides after kernel interface changes.

### Apple Silicon

Build with `cumetal` instead of `gpu`. CuMetal supports the four original search
modes and the numbered self-tests. Supply `--ptx FILE` with `cumetalc` available,
or `--module-dir DIR` containing compiled Metal modules and ABI sidecars. Use
`--help` for runtime library and launch options. RSA/P-256 search commands are
not exposed by this backend.

`gpu` and `cumetal` are mutually exclusive; do not use `--all-features`.

## Search

```sh
./target/release/vanity-miner solana-vanity aaa ""
./target/release/vanity-miner bitcoin-vanity bc1qqqq ""
./target/release/vanity-miner ethereum-vanity 5555 ""
./target/release/vanity-miner shallenge brandonros 000000000000cbaec87e070a04c2eb90644e16f37aab655ccdf683fdda5a6f96
```

Address modes take a prefix and suffix; `""` leaves either unconstrained. Ethereum
requires even-length hex patterns without `0x`.

### RSA and P-256

```sh
./target/release/vanity-miner rsa-modulus-vanity --prefix a --suffix b --private-out rsa-private.pem --public-out rsa-public.pem
./target/release/vanity-miner p256-public-key-vanity --prefix a --target xy --private-out p256-private.pem --public-out p256-public.pem --public-format spki-pem

printf 'example:00000000' > message.bin
./target/release/vanity-miner rsa-pss-signature-vanity --key rsa-private.pem --message message.bin --prefix 0 --signature-out rsa-signature.bin --salt-out rsa-salt.bin
./target/release/vanity-miner p256-signature-vanity --key p256-private.pem --message message.bin --nonce-offset 8 --nonce-length 8 --prefix a --signature-out p256-signature.bin --message-out winning-message.bin --der-out p256-signature.der
```

These four modes accept case-insensitive hex prefixes and suffixes, including odd
lengths, without `0x`. They stop at the first independently verified winner.
`--threads N` selects CPU workers; Ctrl-C cancels between work batches.

| Mode | Pattern applies to |
| --- | --- |
| RSA modulus | 256-byte big-endian modulus, exactly 2048 bits, exponent 65537 |
| RSA-PSS | 256-byte signature, not its hash or salt |
| P-256 public key | `xy` (default), `x`, `y`, or `uncompressed` SEC1 bytes |
| P-256 signature | `raw` (default: 32-byte r then 32-byte s), `r`, or `s` |

- **RSA modulus:** constructs a constrained q progression instead of rejecting
  complete random keys. Winners receive primality, factor-distance, and key checks.
- **RSA-PSS:** uses SHA-256/MGF1-SHA-256 and searches salts by default.
  `--salt-length` defaults to 32, with a maximum of 222. Message-window search
  uses `--search-source message`; see command help for window and fixed-salt options.
- **P-256 signatures:** message-window search uses RFC 6979. `--s-form` selects
  `low` (default), `high`, or `either`; some consumers reject high-S signatures.
  `--search-source ephemeral` keeps the message fixed and searches secret nonces.
  Nonces are never exported: reusing an ECDSA nonce across messages can expose the key.

Private keys are PKCS#8 PEM. RSA public keys are SPKI PEM; P-256 public keys default
to uncompressed SEC1, with SPKI PEM available through `--public-format spki-pem`.
Private files use Unix mode 0600. Existing outputs require `--force`; publication
is atomic per file, not across companion files.

CUDA signing puts private keys in device memory. RSA device operations are
unblinded and checked with the public exponent; host verification uses blinded
RSA. Secret buffers are cleared after synchronization, but complete device-memory
erasure is not guaranteed.

## Self-tests

```sh
cargo build -p vanity-miner --release --locked --no-default-features --features self_test
./target/release/vanity-miner self-test
```

`self_test` enables every mode's logic dependencies, but not its CLI search command.
The same command runs on the backend selected at build time.

All 157 numbered checks and fixtures live in `logic/src/self_test.rs`, with one
kernel per slot in `kernels/src/self_test.rs`. CPU runs report 157 passes and skip
the additional GPU launch probe.

| Slots | Coverage |
| --- | --- |
| 0–117 | Original primitives, pipelines, and compiler regressions |
| 118–125 | P-256 key derivation, points, encoding, invalid scalars |
| 126–134 | P-256 signing, RFC 6979, nonce validity, S forms, message carry |
| 135–144 | SHA-256, MGF1, PSS salt boundaries, CRT and rejection |
| 145–152 | RSA multiplication, progression, primality, boundaries |
| 153–156 | Full RSA/P-256 candidate pipelines against fixed CPU references |

These kernels test candidate logic, not production CUDA argument passing or batch
buffer layouts. CPU passes and successful CUDA compilation do not establish GPU
numerical correctness.
