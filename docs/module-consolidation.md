# Rust module consolidation

This records the earlier source consolidation. Kernel builds have since moved
to self-contained `kernels/<mode>/` and `kernels/self-test-<mode>/` packages,
with GPU sources under `device/src/`
and a build script in each package. The file disposition table below preserves
the historical moves; see the README for current kernel build commands.

## Ownership rules

- `cli/src/modes/<mode>` owns arguments, search state, algorithms, verification, output, and backend-specific kernel sequences.
- `cli/src/runner` owns sessions, workers, progress, and backend resource management. Feature dispatch is explicit; execution helpers do not implement mode algorithms.
- `cli/src/args` owns CLI grammar and genuinely shared argument parsing.
- `logic` owns device-compatible algorithms, cryptographic primitives, encodings, candidate protocols, and executable known-answer checks.
- `kernels` owns exported device entry points. Feature flags, entry-point symbols, result slots, and buffer layouts remain stable.
- Unit tests live with their owner; independent interoperability tests live in `cli/tests`. Production self-tests are a command, separate from test-only helpers.
- Merge thin wrappers and setup-only fragments. New files require a distinct responsibility, not a line-count target.

## Implemented layout

- CLI root: executable/library wiring and application startup.
- `args/`: command selection and shared pattern arguments. Each mode's `args.rs` owns its fields, validation, and description.
- `modes/<mode>/`: one implementation tree. CPU entry points and algorithms share `cpu.rs`; setup lives with mode state. `device.rs` contains the host search shared by CUDA and CuMetal. `cuda.rs` owns that mode's CUDA kernel sequence.
- `modes/rsa_keys.rs`: shared host RSA key validation/encoding. `modes/self_test/`: the production self-test command. `modes/tests.rs`: test-only mode helpers.
- `runner/`: sessions, batches, progress, workers, CUDA and CuMetal resources. RSA stage statistics are owned by the RSA modulus pipeline; progress accepts its formatter.
- `logic/modes/` and kernel packages use the same eight modes as the CLI. Kernel self-test entry points live under `kernels/self-test-<mode>/device/src/lib.rs`.

The original 188 Rust files are accounted for below. Consolidation leaves 177 Rust source files. The nested host `search/` trees, `common/`, and transitional root helper modules are removed.

## File disposition

Paths below are relative to the repository. The left column records the source layout at the start of this consolidation.

| Source | Destination | Responsibility / action |
|---|---|---|
| `cli/build.rs` | `cli/build.rs` | Keep crate entry point or build infrastructure; update wiring |
| `cli/src/application.rs` | `cli/src/application.rs` | Keep crate entry point or build infrastructure; update wiring |
| `cli/src/args.rs` | `cli/src/args/mod.rs` | CLI grammar and dispatch; move per-mode fields, validation and descriptions to each mode’s args.rs |
| `cli/src/common/cpu_workers.rs` | `cli/src/runner/workers/cpu.rs` | CPU worker launch and lifecycle |
| `cli/src/common/cuda_module.rs` | `cli/src/runner/cuda/module.rs` | CUDA module loading |
| `cli/src/common/gpu_context.rs` | `cli/src/runner/cuda/context.rs` | CUDA context and stream ownership |
| `cli/src/common/mod.rs` | Removed | Remove catchall and glob reexports |
| `cli/src/common/pattern_args.rs` | `cli/src/args/pattern.rs` | Shared command-line pattern fields |
| `cli/src/common/search_session.rs` | `cli/src/runner/session.rs`; estimate in `runner/progress.rs` | Merge lifecycle and process interrupt handling with session state |
| `cli/src/common/validation.rs` | `cli/src/modes/{solana,bitcoin}/args.rs; args/mod.rs` | Assign address-specific validation to modes; share hexadecimal parsing |
| `cli/src/device_pipeline.rs` | `cli/src/runner/batches.rs` | Merge producer/consumer delivery; move serialized printing to progress |
| `cli/src/device_workers.rs` | `cli/src/runner/workers/device.rs` | Device thread initialization and lifecycle |
| `cli/src/kernel_modules.rs` | `cli/src/runner/modules.rs` | Host mapping of exported kernel names to artifact names |
| `cli/src/lib.rs` | `cli/src/lib.rs` | Keep crate entry point or build infrastructure; update wiring |
| `cli/src/main.rs` | `cli/src/main.rs` | Keep crate entry point or build infrastructure; update wiring |
| `cli/src/modes/bitcoin/cpu.rs` | `cli/src/modes/bitcoin/cpu.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/bitcoin/cuda.rs` | `cli/src/modes/bitcoin/cuda.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/bitcoin/cumetal.rs` | `cli/src/modes/bitcoin/cumetal.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/bitcoin/mod.rs` | `cli/src/modes/bitcoin/mod.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/ethereum/cpu.rs` | `cli/src/modes/ethereum/cpu.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/ethereum/cuda.rs` | `cli/src/modes/ethereum/cuda.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/ethereum/cumetal.rs` | `cli/src/modes/ethereum/cumetal.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/ethereum/mod.rs` | `cli/src/modes/ethereum/mod.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/mod.rs` | `cli/src/modes/mod.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/p256_public_key/args.rs` | `cli/src/modes/p256_public_key/args.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/p256_public_key/cpu.rs` | `cli/src/modes/p256_public_key/cpu.rs` | Merge CLI entry and CPU implementation in one module |
| `cli/src/modes/p256_public_key/cuda.rs` | `cli/src/modes/p256_public_key/cuda.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/p256_public_key/cumetal.rs` | `cli/src/modes/p256_public_key/cumetal.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/p256_public_key/mod.rs` | `cli/src/modes/p256_public_key/mod.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/p256_public_key/search/cpu.rs` | `cli/src/modes/p256_public_key/cpu.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/p256_public_key/search/device.rs` | `cli/src/modes/p256_public_key/device.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/p256_public_key/search/mod.rs` | `cli/src/modes/p256_public_key/mod.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/p256_public_key/search/prepare.rs` | `cli/src/modes/p256_public_key/mod.rs` | Merge setup with mode state |
| `cli/src/modes/p256_public_key/search/tests.rs` | `cli/src/modes/p256_public_key/tests.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/p256_public_key/search/winner.rs` | `cli/src/modes/p256_public_key/verification.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/p256_signature/args.rs` | `cli/src/modes/p256_signature/args.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/p256_signature/cpu.rs` | `cli/src/modes/p256_signature/cpu.rs` | Merge CLI entry and CPU implementation in one module |
| `cli/src/modes/p256_signature/cuda.rs` | `cli/src/modes/p256_signature/cuda.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/p256_signature/cumetal.rs` | `cli/src/modes/p256_signature/cumetal.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/p256_signature/mod.rs` | `cli/src/modes/p256_signature/mod.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/p256_signature/search/cpu.rs` | `cli/src/modes/p256_signature/cpu.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/p256_signature/search/device.rs` | `cli/src/modes/p256_signature/device.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/p256_signature/search/mod.rs` | `cli/src/modes/p256_signature/mod.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/p256_signature/search/prepare.rs` | `cli/src/modes/p256_signature/mod.rs` | Merge setup with mode state |
| `cli/src/modes/p256_signature/search/tests.rs` | `cli/src/modes/p256_signature/tests.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/p256_signature/search/winner.rs` | `cli/src/modes/p256_signature/verification.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_modulus/args.rs` | `cli/src/modes/rsa_modulus/args.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/rsa_modulus/cpu.rs` | `cli/src/modes/rsa_modulus/cpu.rs` | Merge CLI entry and CPU implementation in one module |
| `cli/src/modes/rsa_modulus/cuda.rs` | `cli/src/modes/rsa_modulus/cuda.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/rsa_modulus/cumetal.rs` | `cli/src/modes/rsa_modulus/cumetal.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/rsa_modulus/mod.rs` | `cli/src/modes/rsa_modulus/mod.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/rsa_modulus/search/constraints.rs` | `cli/src/modes/rsa_modulus/constraints.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_modulus/search/cpu.rs` | `cli/src/modes/rsa_modulus/cpu.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_modulus/search/device.rs` | Removed; `cli/src/modes/rsa_modulus/pipeline.rs` and backend adapters | Both device backends use persistent RSA stages |
| `cli/src/modes/rsa_modulus/search/mod.rs` | `cli/src/modes/rsa_modulus/mod.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_modulus/search/pipeline.rs` | `cli/src/modes/rsa_modulus/pipeline.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_modulus/search/tests.rs` | `cli/src/modes/rsa_modulus/tests.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_pss/args.rs` | `cli/src/modes/rsa_pss/args.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/rsa_pss/cpu.rs` | `cli/src/modes/rsa_pss/cpu.rs` | Merge CLI entry and CPU implementation in one module |
| `cli/src/modes/rsa_pss/cuda.rs` | `cli/src/modes/rsa_pss/cuda.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/rsa_pss/cumetal.rs` | `cli/src/modes/rsa_pss/cumetal.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/rsa_pss/mod.rs` | `cli/src/modes/rsa_pss/mod.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/rsa_pss/search/cpu.rs` | `cli/src/modes/rsa_pss/cpu.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_pss/search/device.rs` | `cli/src/modes/rsa_pss/device.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_pss/search/mod.rs` | `cli/src/modes/rsa_pss/mod.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_pss/search/prepare.rs` | `cli/src/modes/rsa_pss/mod.rs` | Merge setup with mode state |
| `cli/src/modes/rsa_pss/search/tests.rs` | `cli/src/modes/rsa_pss/tests.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/rsa_pss/search/winner.rs` | `cli/src/modes/rsa_pss/verification.rs` | Mode owns its algorithm, result verification, and tests |
| `cli/src/modes/self_test.rs` | `cli/src/modes/self_test/{cpu,cuda}.rs` | Backend execution belongs to the self-test command |
| `cli/src/modes/shallenge/cpu.rs` | `cli/src/modes/shallenge/cpu.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/shallenge/cuda.rs` | `cli/src/modes/shallenge/cuda.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/shallenge/cumetal.rs` | `cli/src/modes/shallenge/cumetal.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/shallenge/mod.rs` | `cli/src/modes/shallenge/mod.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/shallenge/shared_best_hash.rs` | `cli/src/modes/shallenge/shared_best_hash.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/solana/cpu.rs` | `cli/src/modes/solana/cpu.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/solana/cuda.rs` | `cli/src/modes/solana/cuda.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/solana/cumetal.rs` | `cli/src/modes/solana/cumetal.rs` | Keep under its mode; update imports and declarations |
| `cli/src/modes/solana/mod.rs` | `cli/src/modes/solana/mod.rs` | Keep under its mode; update imports and declarations |
| `cli/src/rsa_host.rs` | `cli/src/modes/rsa_keys.rs` | Shared RSA key validation and fixed-width encoding; remove unused CRT conversion |
| `cli/src/rsa_interop.rs` | `cli/tests/rsa_pss_interop.rs` | Independent RSA-PSS compatibility test, compiled only for rsa-pss |
| `cli/src/runner/cpu.rs` | `cli/src/runner/cpu.rs` | Keep backend execution responsibility; update imports |
| `cli/src/runner/cuda_buffers.rs` | `cli/src/runner/cuda/buffers.rs` | Shared persistent device allocations and erasure |
| `cli/src/runner/cuda_transport.rs` | `cli/src/runner/cuda/batch.rs` | Consolidate the persistent candidate transport and its newer replacement under one owner |
| `cli/src/runner/cumetal/address_transport.rs` | Removed; `cli/src/runner/cumetal/batch_transport.rs` | All candidate modes now share structured results and persistent buffers |
| `cli/src/runner/cumetal/batch_transport.rs` | `cli/src/runner/cumetal/batch_transport.rs` | Keep backend execution responsibility; update imports |
| `cli/src/runner/cumetal/driver.rs` | `cli/src/runner/cumetal/driver.rs` | Keep backend execution responsibility; update imports |
| `cli/src/runner/cumetal/mod.rs` | `cli/src/runner/cumetal/mod.rs` | Keep backend execution responsibility; update imports |
| `cli/src/runner/cumetal/module.rs` | `cli/src/runner/cumetal/module.rs` | Keep backend execution responsibility; update imports |
| `cli/src/runner/cumetal/options.rs` | `cli/src/runner/cumetal/options.rs` | Keep backend execution responsibility; update imports |
| `cli/src/runner/cumetal/self_test.rs` | `cli/src/modes/self_test/cumetal.rs` | CuMetal execution belongs to the self-test command |
| `cli/src/runner/gpu.rs` | `cli/src/runner/cuda/mod.rs` | CUDA command dispatch beside CUDA resources |
| `cli/src/runner/mod.rs` | `cli/src/runner/mod.rs` | Keep backend execution responsibility; update imports |
| `cli/src/runner/prepared_cuda.rs` | `cli/src/runner/cuda/batch.rs` | Persistent candidate kernel invocation |
| `cli/src/runner/rsa_pipeline.rs` | `cli/src/modes/rsa_modulus/cuda.rs` | Merge RSA-specific kernel stages with its CUDA adapter |
| `cli/src/search_batches.rs` | `cli/src/runner/batches.rs` | Shared candidate scheduling and verified result delivery |
| `cli/src/search_control.rs` | `cli/src/runner/session.rs` | Session state, cancellation, unique candidate allocation; remove RSA metrics |
| `cli/src/self_test_suite.rs` | `cli/src/modes/self_test/mod.rs` | Production self-test inventory, validation and reporting |
| `cli/src/stats.rs` | `cli/src/runner/progress.rs` | Shared progress and serialized output; RSA stage ownership moves to mode |
| `cli/src/test_support.rs` | `cli/src/modes/tests.rs` | Explicit test-only mock evaluator and output assertions |
| `cli/src/worker_results.rs` | `cli/src/runner/workers/mod.rs` | Shared scoped joining and first-failure handling |
| `cli/tests/rsa_device_pipeline.rs` | `cli/tests/rsa_device_pipeline.rs` | Keep independent integration test; update imports |
| `kernels/src/bitcoin_vanity.rs` | `kernels/src/bitcoin.rs` | Same mode name across crates; exported kernel symbols stay stable |
| `kernels/src/codegen_repros.rs` | `kernels/src/codegen_repros.rs` | Keep crate entry point or build infrastructure; update wiring |
| `kernels/src/ethereum_vanity.rs` | `kernels/src/ethereum.rs` | Same mode name across crates; exported kernel symbols stay stable |
| `kernels/src/lib.rs` | `kernels/src/lib.rs` | Keep crate entry point or build infrastructure; update wiring |
| `kernels/src/match_handler.rs` | `kernels/src/match_handler.rs` | Keep crate entry point or build infrastructure; update wiring |
| `kernels/src/p256_public_key_vanity.rs` | `kernels/src/p256_public_key.rs` | Same mode name across crates; exported kernel symbols stay stable |
| `kernels/src/p256_signature_vanity.rs` | `kernels/src/p256_signature.rs` | Same mode name across crates; exported kernel symbols stay stable |
| `kernels/src/rsa_modulus_vanity.rs` | `kernels/src/rsa_modulus.rs` | Same mode name across crates; exported kernel symbols stay stable |
| `kernels/src/rsa_pss_signature_vanity.rs` | `kernels/src/rsa_pss.rs` | Same mode name across crates; exported kernel symbols stay stable |
| `kernels/src/self_test_bitcoin.rs` | `kernels/src/self_test/bitcoin.rs` | Self-test entry points grouped by command, one feature per mode |
| `kernels/src/self_test_ethereum.rs` | `kernels/src/self_test/ethereum.rs` | Self-test entry points grouped by command, one feature per mode |
| `kernels/src/self_test_p256_public_key.rs` | `kernels/src/self_test/p256_public_key.rs` | Self-test entry points grouped by command, one feature per mode |
| `kernels/src/self_test_p256_signature.rs` | `kernels/src/self_test/p256_signature.rs` | Self-test entry points grouped by command, one feature per mode |
| `kernels/src/self_test_rsa_modulus.rs` | `kernels/src/self_test/rsa_modulus.rs` | Self-test entry points grouped by command, one feature per mode |
| `kernels/src/self_test_rsa_pss.rs` | `kernels/src/self_test/rsa_pss.rs` | Self-test entry points grouped by command, one feature per mode |
| `kernels/src/self_test_shallenge.rs` | `kernels/src/self_test/shallenge.rs` | Self-test entry points grouped by command, one feature per mode |
| `kernels/src/self_test_solana.rs` | `kernels/src/self_test/solana.rs` | Self-test entry points grouped by command, one feature per mode |
| `kernels/src/shallenge.rs` | `kernels/src/shallenge.rs` | Keep crate entry point or build infrastructure; update wiring |
| `kernels/src/solana_vanity.rs` | `kernels/src/solana.rs` | Same mode name across crates; exported kernel symbols stay stable |
| `kernels/tests/self_test_slots.rs` | `kernels/tests/self_test_slots.rs` | Keep independent integration test; update imports |
| `logic/src/crypto/ed25519.rs` | `logic/src/crypto/ed25519.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/keccak256.rs` | `logic/src/crypto/keccak256.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/mod.rs` | `logic/src/crypto/mod.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/p256/mod.rs` | `logic/src/crypto/p256/mod.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/p256/signatures.rs` | `logic/src/crypto/p256/signatures.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/ripemd160.rs` | `logic/src/crypto/ripemd160.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/rsa_crt.rs` | `logic/src/crypto/rsa_crt.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/rsa_prime.rs` | `logic/src/crypto/rsa_prime.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/rsa_pss.rs` | `logic/src/crypto/rsa_pss.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/secp256k1.rs` | `logic/src/crypto/secp256k1.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/sha256.rs` | `logic/src/crypto/sha256.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/crypto/sha256_digest.rs` | `logic/src/crypto/sha256_digest.rs` | Keep the RustCrypto Digest trait adapter; the SHA-256 algorithm remains in sha256.rs |
| `logic/src/crypto/sha512.rs` | `logic/src/crypto/sha512.rs` | Keep: named cryptographic primitive or adapter |
| `logic/src/encoding/base58.rs` | `logic/src/encoding/base58.rs` | Keep: named encoding |
| `logic/src/encoding/bech32.rs` | `logic/src/encoding/bech32.rs` | Keep: named encoding |
| `logic/src/encoding/mod.rs` | `logic/src/encoding/mod.rs` | Keep: named encoding |
| `logic/src/lib.rs` | `logic/src/lib.rs` | Keep crate entry point or build infrastructure; update wiring |
| `logic/src/modes/bitcoin_vanity.rs` | `logic/src/modes/bitcoin.rs` | Same mode name across crates; device-compatible implementation |
| `logic/src/modes/ethereum_vanity.rs` | `logic/src/modes/ethereum.rs` | Same mode name across crates; device-compatible implementation |
| `logic/src/modes/mod.rs` | `logic/src/modes/mod.rs` | Keep under its mode; update imports and declarations |
| `logic/src/modes/p256_public_key_vanity.rs` | `logic/src/modes/p256_public_key.rs` | Same mode name across crates; device-compatible implementation |
| `logic/src/modes/p256_signature_vanity.rs` | `logic/src/modes/p256_signature.rs` | Same mode name across crates; device-compatible implementation |
| `logic/src/modes/rsa_modulus_vanity.rs` | `logic/src/modes/rsa_modulus.rs` | Same mode name across crates; device-compatible implementation |
| `logic/src/modes/rsa_modulus_vanity/pipeline.rs` | `logic/src/modes/rsa_modulus.rs` | Mode-owned fixed-width device pipeline |
| `logic/src/modes/rsa_pss_signature_vanity.rs` | `logic/src/modes/rsa_pss.rs` | Same mode name across crates; device-compatible implementation |
| `logic/src/modes/shallenge.rs` | `logic/src/modes/shallenge.rs` | Keep under its mode; update imports and declarations |
| `logic/src/modes/solana_vanity.rs` | `logic/src/modes/solana.rs` | Same mode name across crates; device-compatible implementation |
| `logic/src/search/candidate_derivation.rs` | `logic/src/search/candidate_derivation.rs` | Keep: device-compatible candidate protocol, derivation, or matching primitive |
| `logic/src/search/candidate_result.rs` | `logic/src/search/candidate_result.rs` | Keep: device-compatible candidate protocol, derivation, or matching primitive |
| `logic/src/search/device_record.rs` | `logic/src/search/device_record.rs` | Keep: device-compatible candidate protocol, derivation, or matching primitive |
| `logic/src/search/hex_pattern.rs` | `logic/src/search/hex_pattern.rs` | Keep: device-compatible candidate protocol, derivation, or matching primitive |
| `logic/src/search/message_window.rs` | `logic/src/search/message_window.rs` | Keep: device-compatible candidate protocol, derivation, or matching primitive |
| `logic/src/search/mod.rs` | `logic/src/search/mod.rs` | Keep: device-compatible candidate protocol, derivation, or matching primitive |
| `logic/src/search/salt_counter.rs` | `logic/src/search/salt_counter.rs` | Keep: device-compatible candidate protocol, derivation, or matching primitive |
| `logic/src/search/vanity.rs` | `logic/src/search/vanity.rs` | Keep: device-compatible candidate protocol, derivation, or matching primitive |
| `logic/src/search/xoroshiro.rs` | `logic/src/search/xoroshiro.rs` | Keep: device-compatible candidate protocol, derivation, or matching primitive |
| `logic/src/self_test/bitcoin/base58_probes.rs` | `logic/src/self_test/bitcoin/base58_probes.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/bitcoin/cases.rs` | `logic/src/self_test/bitcoin/cases.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/bitcoin/layout_probes.rs` | `logic/src/self_test/bitcoin/layout_probes.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/bitcoin/mod.rs` | `logic/src/self_test/bitcoin/mod.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/bitcoin/secp256k1_probes.rs` | `logic/src/self_test/bitcoin/secp256k1_probes.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/ethereum/cases.rs` | `logic/src/self_test/ethereum/cases.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/ethereum/mod.rs` | `logic/src/self_test/ethereum/mod.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/known_answers.rs` | `logic/src/self_test/known_answers.rs` | Keep the four constants shared by multiple self-test modes; mode-specific fixtures remain with their checks |
| `logic/src/self_test/metadata.rs` | `logic/src/self_test/metadata.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/mod.rs` | `logic/src/self_test/mod.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/p256_public_key/cases.rs` | `logic/src/self_test/p256_public_key/cases.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/p256_public_key/fixtures.rs` | `logic/src/self_test/p256_public_key/fixtures.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/p256_public_key/mod.rs` | `logic/src/self_test/p256_public_key/mod.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/p256_signature/cases.rs` | `logic/src/self_test/p256_signature/cases.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/p256_signature/fixtures.rs` | `logic/src/self_test/p256_signature/fixtures.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/p256_signature/mod.rs` | `logic/src/self_test/p256_signature/mod.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/rsa_modulus/cases.rs` | `logic/src/self_test/rsa_modulus/cases.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/rsa_modulus/fixtures.rs` | `logic/src/self_test/rsa_modulus/fixtures.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/rsa_modulus/mod.rs` | `logic/src/self_test/rsa_modulus/mod.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/rsa_pss/cases.rs` | `logic/src/self_test/rsa_pss/cases.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/rsa_pss/fixtures.rs` | `logic/src/self_test/rsa_pss/fixtures.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/rsa_pss/mod.rs` | `logic/src/self_test/rsa_pss/mod.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/shallenge/cases.rs` | `logic/src/self_test/shallenge/cases.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/shallenge/mod.rs` | `logic/src/self_test/shallenge/mod.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/solana/arithmetic.rs` | `logic/src/self_test/solana/arithmetic.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/solana/base58_probes.rs` | `logic/src/self_test/solana/base58_probes.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/solana/bisect_scalar52.rs` | `logic/src/self_test/solana/bisect_scalar52.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/solana/cases.rs` | `logic/src/self_test/solana/cases.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/solana/ed25519_probes.rs` | `logic/src/self_test/solana/ed25519_probes.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/solana/layout_probes.rs` | `logic/src/self_test/solana/layout_probes.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |
| `logic/src/self_test/solana/mod.rs` | `logic/src/self_test/solana/mod.rs` | Keep: mode checks, case inventory, fixtures, or focused arithmetic probe |

## Removal of superseded execution paths

- Device workers execute once for their lifetime; the winner-round coordinator is removed.
- Candidate modes use one evaluation and verification path. First-match library calls
  stop after verification; continuous device sessions overlap launches and verification.
- CUDA and CuMetal share structured candidate results for addresses, nonces, P-256,
  and RSA-PSS. Address/nonce output is reconstructed on the CPU before publication.
- RSA modulus uses the same persistent four-stage pipeline on both device backends.
  The old batch request, host device constructor, and versioned entry points are removed.
  CPU modulus construction and independent arithmetic references remain.
- Both LLVM 7 and LLVM 21 build configurations remain supported.

Rebuild host binaries, PTX overrides, and CuMetal artifacts together after these ABI changes.
CuMetal's RSA `--batches` limit counts pipeline cycles, each containing four kernel launches.
Match statistics count verified exported records; a batch can contain additional matching lanes.

`BATCH_SIZE` now controls all CUDA candidate batches. It replaces `CRYPTO_BATCH_SIZE`;
the old `BLOCKS_PER_SM` address-only launch setting is removed.

## Validation after removal

- Aggregate CPU and CuMetal compilation passed. All 16 individual production-mode
  CPU/CuMetal combinations passed `cargo check --all-targets`.
- Host tests: 49 CLI library tests, 8 RSA pipeline integration tests, and 1 independent
  RSA-PSS interoperability test passed. The CuMetal configuration passed 48 library tests.
- Shared logic: 78 tests passed. The actual CPU CLI `self-test` command passed all 160 checks.
- All eight kernel modes passed host compilation; the kernel self-test slot and guard test passed.
- The Linux Lima LLVM 21 release check compiled all eight CUDA production PTX modules
  and eight self-test PTX modules. LLVM 7 configuration is retained; it was not rebuilt here.
- CUDA runtime execution remains unverified because the Lima builder has no NVIDIA GPU.
- CuMetal runtime execution remains unverified. The installed Nix compiler rejects
  `shf.l.wrap.b32` in the generated Shallenge and RSA PTX. The installed Apple compiler
  rejects pointer subtraction in the Shallenge PTX. These attempts did not reach kernel execution.
  Those existing binaries had unverified source revisions; these failures do not
  establish the status of the current CuMetal contribution series. See the
  [validation provenance rules](../AGENTS.md) before repeating the checks.
- Root and kernel workspace formatting, shell syntax, and `git diff --check` passed.

## Historical pinned CuMetal validation

The following records the earlier `98cf505` validation. See the
[current validation report](cumetal-validation.md) for the later tested pin.

At that time, the CuMetal flake input pinned the cumulative fork contribution revision
`98cf50573d091162a507e83f1d9e07e45b477ed1`, including its VF64 submodule. The Nix
package builds the compiler and runtime together. The CLI embeds the locked
revision and verifies the package manifest and both hashes before loading it.
Independent binary selection and precompiled Metal input are removed; PTX is
snapshotted and translated by the verified compiler on each run.

Validation on Apple Silicon:

- The Nix package built successfully from the locked source. Its final artifact
  hashes match the generated manifest.
- All 51 CuMetal host library tests passed, including wrong-revision, replaced
  compiler/runtime, and missing-manifest rejection. Aggregate CPU checking passed.
- The real CLI rejects the old local build directory and retired override flags
  before device execution. Linux `default`, `v7`, and `v21` shells remain available.
- The documented Nix-shell invocation ran one 32-candidate Shallenge GPU batch
  with seed 1, username `miner`, an all-ones target, and `--verify`. One verified
  nonce was exported; every candidate and the match count agreed with the CPU.
- The PTX hash matches the final LLVM 21 Shallenge artifact from the cleanup build
  committed as vanity-miner `a2aba42`. Other modes were not run in this validation.

| Artifact | SHA-256 |
| --- | --- |
| Compiler | `83b6f09816b5b68c56af230b90372ae7a588c85bb6ab7fe0f3f167226b301b7d` |
| Runtime | `2c820652ebb7f129b61d0c01bbeaf0ae974304c7a8fe1ef5a193d65f3079a916` |
| Shallenge PTX | `e39509140aaf2cb8272729c2888790788d6c24edcd2b65489fa19d55b1a042c3` |

Package output: `/nix/store/wrgw94blr7h6p761zv4yj4jwwv3fwjy2-vanity-cumetal-98cf50573d09`.
