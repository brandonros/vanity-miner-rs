# Four cryptographic vanity modes

Implemented on branch `new-modes` in the `vanity-miner-new-modes` worktree,
starting from `poc/llvm21-portable-ptx` (`6a5b9e6`).

## Scope and implementation

All four commands have CPU runners, substantive CUDA kernels, CUDA host dispatch,
feature forwarding, shared candidate logic, independent winner reconstruction,
protected output, cancellation, statistics, and documentation:

| Mode | Device work | Host setup and verification |
| --- | --- | --- |
| RSA modulus | q progression, modulus multiplication/matching, small-prime sieve and 32-base Miller-Rabin filter | Random feasible p and q interval/residue setup; independent randomized primality, component and sign/verify checks |
| RSA-PSS signature | Salt/message enumeration, SHA-256, MGF1, explicit-salt PSS, fixed-width CRT private operation, e=65537 fault check, raw signature matching | Validate input key; reproduce with blinded RSA implementation; independent PSS verification |
| P-256 public key | HMAC candidate derivation, rejection sampling, scalar multiplication, point matching | Re-derive scalar/point and validate encoding before key export |
| P-256 signature | RFC6979 message-window or domain-separated ephemeral search, scalar arithmetic, exact r/s/raw matching and S-form selection | Reconstruct candidate, compare exact signature bytes and verify |

Each mode has matching `logic/src/` and `kernels/src/` modules:
`rsa_modulus_vanity.rs`, `rsa_pss_signature_vanity.rs`,
`p256_public_key_vanity.rs`, and `p256_signature_vanity.rs`. Each logic module
owns its request record and candidate evaluator; each kernel module exports one
CUDA entry point. `logic/src/candidate_result.rs` holds the shared result record. `cli/src/runner/cuda_batches.rs` transports bounded batches, synchronizes results,
and clears secret request/result buffers before release. The shared CUDA runner starts one worker per device, reserves only one verified
winner across workers, and shares the CPU runners' final reconstruction and
output code. Contexts, modules, and streams are owned by the common GPU context. It does not call a
CPU search as a GPU fallback.

The initial GPU transport uses conservative 64-candidate batches and a 64-KiB
per-thread stack limit (`STACK_SIZE` overrides it). These are unmeasured starting
settings, not performance claims. Occupancy, CRT setup caching and batch tuning should follow hardware validation.

The user explicitly deferred Vast.ai execution and Rust-CUDA compiler bug fixing
to a later session. No VM or GPU rental was provisioned. PTX compilation and
on-device proof remain deferred; source integration and host/shared tests are
complete for this phase.

## Validation evidence

- Every new CPU feature checks individually and all four check together.
- Every new kernel feature checks individually and all four check together on
  native Darwin arm64. This checks Rust types and feature gates; it is **not** a
  PTX build or GPU execution.
- Combined existing/new CPU release tests: 64 shared-logic tests and 18 host tests
  passed before the final ABI-layout test was added. The ABI test and corrupted
  device-result rejection also passed separate focused runs.
- Host integration tests run each mode through both its CPU search and a test
  implementation executing the exact device candidate functions on the host.
  Both signature sources, output reconstruction, finite exhaustion, and key
  export are covered. These tests do not claim physical CPU/GPU parity.
- Independent checks include RFC6979 known answers, Python-derived HMAC/MGF1/PSS
  vectors, RustCrypto RSA verification using separate bigint arithmetic, and
  LibreSSL key/signature verification. Fresh RSA test keys remain ephemeral.
- All four release CLI commands passed smoke tests, including both signature
  sources. Ctrl-C left no output files for a cancelled P-256 search. Private
  output permissions were checked as 0600.
- New-mode Clippy passes with warnings denied. Full existing/new Clippy reports
  pre-existing lints in base58, bech32, SHA-256/SHA-512 macro expansions and other
  legacy modules. These unrelated implementations were not rewritten.
- The existing 118-check CPU self-test and all four crypto fixture groups passed.
  `self-test` now also runs
  enabled crypto differential fixtures: all P-256 targets/S forms, counter and
  salt carries, RSA CRT/PSS salt boundaries, progression bounds and primality.
  GPU builds run these fixtures through the real CUDA transport on each device.
- Formatting is applied to changed/new code. Whole-workspace rustfmt encounters
  existing formatting differences and a trailing-whitespace error in the legacy
  Ethereum module; broad formatting-only changes were discarded.

## Dependencies and deferred build checks

The shared implementation uses existing no_std configurations of `p256` 0.13.2,
`ecdsa` 0.16.9, `crypto-bigint` 0.5.5, SHA-256 and HMAC. No replacement elliptic
curve or multiprecision integer library was necessary. RSA PEM/key generation and
independent validation stay in the host crate. Both workspace lockfiles include
the new dependencies.

Local Rust is stable 1.97.1 on Apple Silicon, with no CUDA tools. The full GPU
build previously stopped in existing `smallvec` on
`#![feature(dropck_eyepatch)]` (E0554), before kernel compilation. Consequently the
CUDA host transport and PTX build still need the pinned Linux/CUDA toolchain.
No compiler compatibility or GPU throughput claim is made here.

On the later CUDA host, build each new feature with `gpu`, then the combined
features and existing modes. Add `self_test` and run `self-test` before searches:

```sh
cargo build -p vanity-miner --release --no-default-features --features gpu,rsa-modulus,rsa-pss,p256-public-key,p256-signature,self_test --locked
./target/release/vanity-miner self-test
```

Use the existing LLVM/toolchain workflow's `llvm21` selection when appropriate
for the rented hardware. Run compute-sanitizer and cancellation/output smoke
tests there, including repeated matches and multiple devices.

## Security and filesystem limits

Seeds come from OS entropy; PRF inputs separate domains, key fingerprints,
messages, workers and counters. Scalar candidates use rejection sampling. No
private scalar, factor, seed or ephemeral nonce is printed or persisted as search
metadata. Private files use PKCS#8 PEM and Unix mode 0600.

GPU signing places private-key components in device memory. RSA device CRT is
unblinded, checks each result with the public exponent, and is intended for this
local search pipeline rather than a signing oracle. CPU PSS uses blinding.
Explicit secret fields and transport allocations are cleared; compiler-created
copies, registers and failed-device erasure cannot be guaranteed wiped. Erasure
failures are reported without dumping data.

All output files are staged before publication and the private key/raw signature
is published last. Each destination is atomic and refuses overwrite without
`--force`; duplicate output paths and replacement of inputs are rejected. Multiple
arbitrary filesystem paths cannot form one atomic rename: companion output may
remain if a later publication fails, and abrupt termination can leave protected
staging files. This is a documented limitation, not a multi-file atomicity claim.

The CLI uses direct per-mode CPU/GPU entry points under `cli/src/modes/`.
Typed mode callbacks pass candidate batches to CUDA transport. Shared batching
and winner selection live in `cli/src/search_batches.rs`; host evaluation
adapters live in `cli/src/test_support.rs` and only compile for tests/self-test.
There is no cross-mode request enum or device-search trait.

All backends use `self_test_suite` for test inventory and reporting. CPU and CUDA
execute the enabled crypto fixtures; CuMetal reports those entries as unsupported.
Search statistics share `GlobalStats`; `SearchControl` only coordinates bounded
work and winner/cancellation state, forwarding counts to the shared statistics.
