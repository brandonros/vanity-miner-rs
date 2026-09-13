# CUDA Oxide selective backport review

Compared `master` at `20777dc` with `cuda-oxide` at `bb4837f`.
The backport branch starts at master; neither original branch was rebased or squashed.

`logic/src/self_test.rs` is already byte-for-byte identical on both branches
(blob `19d1ad40565f0c360a1edcaf3ff5a1253b4f7a8a`, 118 checks).
It reached master in `3149ffe`. Branch-only commit counts therefore overstate missing work.

## File decisions

Renames are listed as two paths so every changed path has a disposition.

| File | Decision |
|---|---|
| `.github/workflows/cuda-compile.yaml` | Keep Rust-CUDA build, both architectures, and artifacts; cache replacement is separate CI work. |
| `.gitignore` | Skip branch scratch temp/ entry; not needed by selected changes. |
| `Cargo.lock` | Keep Rust-CUDA dependencies, feature configuration, and build toolchain; CUDA Oxide migration. |
| `Cargo.toml` | Keep Rust-CUDA dependencies, feature configuration, and build toolchain; CUDA Oxide migration. |
| `GOAL.md` | Skip CUDA Oxide compiler investigation instructions. |
| `KNOWN_FAILURES.md` | Skip historical CUDA Oxide failure log; not evidence of Rust-CUDA failures. |
| `README.md` | Keep Rust-CUDA build/run instructions and current documentation. |
| `cli/Cargo.toml` | Keep Rust-CUDA dependencies, feature configuration, and build toolchain; CUDA Oxide migration. |
| `cli/build.rs` | Keep Rust-CUDA dependencies, feature configuration, and build toolchain; CUDA Oxide migration. |
| `cli/src/args.rs` | Keep master feature gates, CPU worker naming, and username validation where applicable. |
| `cli/src/common/cpu_workers.rs` | Keep newer cpu_workers name; source is identical to CUDA branch workers.rs. |
| `cli/src/common/gpu_context.rs` | Adapt fallible integer parsing; retain cust context and 16 KiB stack default. |
| `cli/src/common/mod.rs` | Keep master feature gates, CPU worker naming, and username validation where applicable. |
| `cli/src/common/validation.rs` | Keep master feature gates, CPU worker naming, and username validation where applicable. |
| `cli/src/common/workers.rs` | Rename counterpart only; no implementation to backport. |
| `cli/src/modes/bitcoin.rs` | Keep cust runtime and master diagnostics; changes primarily migrate GPU context, buffers, and launches. |
| `cli/src/modes/ethereum.rs` | Keep cust runtime and master diagnostics; changes primarily migrate GPU context, buffers, and launches. |
| `cli/src/modes/mod.rs` | Keep master feature gates, CPU worker naming, and username validation where applicable. |
| `cli/src/modes/self_test.rs` | Keep existing cust launch/reporting. Same 118 checks; smaller stub buffer is unnecessary and would require reviewing the Rust-CUDA kernel slice contract. |
| `cli/src/modes/shallenge.rs` | Keep cust runtime and master diagnostics; changes primarily migrate GPU context, buffers, and launches. |
| `cli/src/modes/solana.rs` | Keep cust runtime and master diagnostics; changes primarily migrate GPU context, buffers, and launches. |
| `cli/src/runner/cpu.rs` | Keep master feature gates, CPU worker naming, and username validation where applicable. |
| `cli/src/runner/gpu.rs` | Keep cust runtime and master diagnostics; changes primarily migrate GPU context, buffers, and launches. |
| `docs/benchmark-2025-07-09.md` | Keep existing documentation; deletion does not improve portable logic. |
| `docs/cuda.md` | Keep existing documentation; deletion does not improve portable logic. |
| `flake.lock` | Keep Rust-CUDA dependencies, feature configuration, and build toolchain; CUDA Oxide migration. |
| `flake.nix` | Keep Rust-CUDA dependencies, feature configuration, and build toolchain; CUDA Oxide migration. |
| `kernels/Cargo.toml` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `kernels/src/atomic.rs` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `kernels/src/bitcoin_vanity.rs` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `kernels/src/ethereum_vanity.rs` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `kernels/src/lib.rs` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `kernels/src/match_handler.rs` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `kernels/src/self_test.rs` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `kernels/src/shallenge.rs` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `kernels/src/solana_vanity.rs` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `kernels/src/utilities.rs` | Keep Rust-CUDA kernel ABI, module layout, feature gates, and atomics; CUDA Oxide migration. |
| `logic/Cargo.toml` | Keep Rust-CUDA dependencies, feature configuration, and build toolchain; CUDA Oxide migration. |
| `logic/src/base58.rs` | Backport leading-zero and all-zero unit tests. |
| `logic/src/bech32.rs` | Keep generic witness encoder; deleting a public helper is unrelated to the backport. |
| `logic/src/bitcoin_vanity.rs` | Backport four WIF vectors and public-key/hash/match assertions. |
| `logic/src/ed25519.rs` | Keep existing precomputed-table call; no demonstrated Rust-CUDA benefit from changing the equivalent API. |
| `logic/src/lib.rs` | Keep master feature gates, CPU worker naming, and username validation where applicable. |
| `logic/src/secp256k1.rs` | Skip ManuallyDrop workaround for CUDA Oxide; retain secret-key drop/zeroization. |
| `logic/src/sha256.rs` | Backport tests and padding cleanup in separate commits; add independent boundary vectors. |
| `logic/src/shallenge.rs` | Backport tests; retain dynamic username implementation and add length coverage. |
| `logic/src/solana_vanity.rs` | Backport intermediate hash and public-key assertions. |
| `logic/src/xoroshiro.rs` | Backport fixed private-key vector, determinism, and alphabet tests. |
| `rust-toolchain.toml` | Keep Rust-CUDA dependencies, feature configuration, and build toolchain; CUDA Oxide migration. |
| `scripts/build-gpu.sh` | Keep Rust-CUDA dependencies, feature configuration, and build toolchain; CUDA Oxide migration. |
| `scripts/vast-run.sh` | Keep current local-binary deployment workflow; remote host and release/PTX workflow changes are separate work. |

## Validation

- Original all-feature logic suite: 23 tests passed.
- Expanded all-feature logic suite: 40 tests passed before and after SHA-256 cleanup.
- Default-feature logic suite: 15 tests passed.
- CPU CLI self-test: all 118 checks passed.
- `git diff --check`: passed.
- Used installed Cargo 1.97.0 / Rust 1.97.1 on aarch64 macOS, not the pinned Linux CUDA nightly. Existing macOS deployment-target linker warnings did not prevent CPU tests.
- GPU check attempted with `cargo check -p vanity-miner --features gpu,self_test --locked --offline`; dependency resolution stopped because `addr2line v0.24.2` was not cached. GPU compilation and execution remain unverified. Validate on the project's Linux CUDA environment before merging the GPU settings change.

## How to integrate

Review `git diff master...backport/cuda-oxide-improvements` and the focused commits.
Merge the branch when ready, or cherry-pick individual commits. Keep the portable test
commit before the SHA-256 cleanup commit. No rebase of cuda-oxide is needed.
If master advances, only update this small backport branch against master.
