# CuMetal validation — 2026-09-15

## Result

Retested **2026-09-15, 18:36:19–18:59:34 UTC**, against CuMetal
**`e5acf8cc0c658142c704ee80e749f5180911fff4`**, matching `flake.lock` at validation
time. All eight production modes and all eight self-test groups were attempted.
The [16-row status table](cumetal-status.md) has **3 true and 13 false**.

The lock later advanced to `92a9b8f4de230199eac617fc57feaf7c0849cf01`, then
`7d12f120a6b80a9956588de2b974db5747b35f57`.
This report preserves the complete `e5acf8cc0c65` run. The
[issue ownership matrix](cumetal-issue-matrix.md) records upstream status and
separately identified targeted research on later revisions.

Production: **2 modes passed, 4 failed translation, 2 timed out**. Both search
sources were tested for each signature mode, giving ten production commands.
Self-tests: **8 passed, 152 failed, 0 skipped**; all 152 failed slots were blocked
by translation in seven modules. No numerical assertion failures were observed.

| Production mode | Result | Fresh evidence |
| --- | --- | --- |
| Solana | Translation failed | `i32` operand refers to an `i64` value; first diagnostic at PTX line 34070. |
| Bitcoin | Translation failed | Pointer subtraction requires a pointer minus a 64-bit integer byte offset; PTX line 26556. |
| Ethereum | Translation failed | Same pointer-subtraction diagnostic; PTX line 22069. |
| Shallenge | Passed bounded GPU check | Two batches, 64 nonces, two CPU-verified outputs; exit 0. |
| P-256 public key | Passed bounded GPU check | Two batches, 64 keys, two CPU-verified outputs; exit 0. |
| P-256 signature | Translation failed | Both message and ephemeral search: `%rd17653` undefined on an incoming edge to `$L__BB0_11`. |
| RSA modulus | Timed out; execution unverified | All four Metal sources and ABI sidecars emitted. A live stack sample showed `cuCtxSynchronize` waiting in `_MTLCommandBuffer waitUntilCompleted`. No completed cycle or tested factor candidates within 300 seconds. |
| RSA-PSS signature | Timed out; execution unverified | Both salt and message search emitted identical 24,263,894-byte Metal sources and ABI sidecars. Each was sampled inside Apple's `newLibraryWithSource` compilation and timed out at 300 seconds with no completed candidates. |

“Eight modes” includes eleven production entry points: RSA modulus has generate,
ranges, search, and advance entries. Emitting their source does not establish that
all four entries executed successfully. Timeouts are not numerical failures and
do not establish that compilation or execution could never finish.

## What changed from the previous run

The historical `98cf505` run also had **3 true / 13 false**, but those totals hide
changes in the failure stages:

- **RSA-PSS production advances past translation.** The former pointer-type
  diagnostic at PTX line 55188 is absent in both fresh production runs. Metal
  source compilation is now the observed wait; GPU correctness remains unverified.
- **RSA modulus still times out**, but this run was directly sampled waiting for
  Metal command completion. The earlier report's compilation-time explanation
  cannot be carried forward as the explanation of this fresh observation.
- **RSA-PSS self-test still fails translation, with a different diagnostic:**
  the first reported location is now PTX line 2377, rather than 358262. This needs
  investigation; a changed diagnostic alone does not establish its root cause or
  prove a regression.
- The other failing modules retain their previous first diagnostics. Shallenge
  production/self-tests and P-256 public-key production pass again.

The unchanged PTX inputs make this a direct before/after test of the CuMetal pin.
Timing differences are not benchmarks or proof of a compiler speedup: shared
Metal caches were not cleared, and other compiler work was running on the Mac.

## Self-tests

Each group ran in its own process, selecting every slot owned by that group with
repeated `--self-test-slot` arguments. Slot filtering does not reduce the kernel
body. In the passing Shallenge group, the full kernel executed and the runner
checked that unrelated result slots retained their sentinel values. The other
seven modules failed before execution. Group selection prevented one slow module
from blocking the remaining checks.

| Self-test group | Passed | Blocked by translation | First diagnostic | Seconds |
| --- | ---: | ---: | --- | ---: |
| Solana | 0 | 72 | Trap-call expansion exceeds bounded control-flow size: 11 calls, 1,746 blocks, 274,910 operations. | 13.06 |
| Bitcoin | 0 | 33 | `ptr<device, i16>` operand refers to an `i64` value; PTX line 53099. | 26.00 |
| Ethereum | 0 | 5 | Trap-call expansion exceeds bounded control-flow size: 5 calls, 372 blocks, 271,425 operations. | 20.67 |
| Shallenge | 8 | 0 | All eight owned slots passed. | 1.70 |
| P-256 public key | 0 | 9 | Trap-call expansion exceeds bounded control-flow size: 8 calls, 1,039 blocks, 274,906 operations. | 14.37 |
| P-256 signature | 0 | 10 | `%rs506` undefined on an incoming edge to `$L__BB0_1`. | 64.39 |
| RSA modulus | 0 | 12 | `%rs2233` undefined on an incoming edge to `$L__BB0_1`. | 88.66 |
| RSA-PSS | 0 | 11 | `ptr<device, i16>` operand refers to an `i64` value; PTX line 2377. | 180.86 |
| **Total** | **8** | **152** | **All 160 selected slots reported exactly once across the eight runs.** | |

The aggregate excludes the `not selected` skips printed for other groups in each
process. It contains no missing slots, duplicate outcomes, or observed skips.
A failed module launch is reported against every slot it owns, so 152 failed
slots represent seven compiler-blocked modules, not 152 independent algorithm
failures.

RSA-PSS slot 155 is configured to skip its disabled end-to-end GPU check when the
module executes. Here compilation failed first, so it was reported as failed
with the other ten slots. A future successful enabled suite would normally report
159 passes and one intentional skip; it must not be described as 160 GPU passes.

## Why included PRs do not make these workloads pass

This run loaded the intended build. The host was rebuilt from a frozen source
snapshot with the current lock. Every command verified and printed the pinned
compiler/runtime hashes. The source audit also matched the immutable importer
and control-flow source files to Git commit `e5acf8cc0c65` and verified relevant
PR commits are ancestors of that revision.

The remaining gap is implementation and validation scope:

| Workload | Included fixes | Remaining gap |
| --- | --- | --- |
| Solana, Bitcoin, Ethereum production | [PR #50](https://github.com/Lulzx/cuda-metal/pull/50) fixes conversion source semantics; [#56](https://github.com/Lulzx/cuda-metal/pull/56) supports valid pointer subtraction; [#75](https://github.com/Lulzx/cuda-metal/pull/75) preserves inferred definition types. | Conversion destinations are still initially inferred with the wrong width before SSA, the compiler's representation of values at control-flow joins. Correcting them later does not repair every earlier use. That required inference change is not implemented. |
| P-256 signature production | [PR #85](https://github.com/Lulzx/cuda-metal/pull/85) and subsequent changes handle bounded constant-predicate paths. | Traversal still stops at empty blocks created by consecutive labels. The needed normalization or traversal extension is not implemented; the control-flow source is unchanged since `98cf505`. |
| RSA-PSS production | [PR #108](https://github.com/Lulzx/cuda-metal/pull/108), [#110](https://github.com/Lulzx/cuda-metal/pull/110), and [#112](https://github.com/Lulzx/cuda-metal/pull/112) are included. | The fresh full PTX now translates, but neither production variant completes Metal compilation and verified GPU execution within the limit. |

The inspected PR descriptions do not claim that the exact retained Solana,
Bitcoin, Ethereum, and P-256 signature artifacts all passed. The descriptions of
PR #75 and PR #85 explicitly identify remaining scope or workload limitations.
Issue ownership, a diagnosed cause, an implemented small regression test, full
PTX translation, and a verified GPU run are separate milestones. A status row is
marked true only after the recorded GPU validation passes.

### Current upstream tracking

The unresolved implementation work is explicit in existing upstream
[#76](https://github.com/Lulzx/cuda-metal/issues/76) (conversion/join types),
[#83](https://github.com/Lulzx/cuda-metal/issues/83) (empty labels and incomplete
predicate rewrites), [#35](https://github.com/Lulzx/cuda-metal/issues/35) (masked
payload demand), and [#46](https://github.com/Lulzx/cuda-metal/issues/46)
(finite-loop helper reuse). The RSA-PSS compilation-time investigation has its
own [#115](https://github.com/Lulzx/cuda-metal/issues/115). The narrower PRs #57,
#62 and #85 reference their expanded issues without automatically closing them.

RSA modulus now has [#119](https://github.com/Lulzx/cuda-metal/issues/119) for its
GPU completion investigation. Exact-binary disassembly resolves the sampled wait
to `kernel_rsa_generate`. Fixed public scalar/native CPU fixtures complete with
matching rejection statuses, but neither the GPU wait's cause nor its correction
is established. This is distinct from the RSA-PSS compilation wait in #115.

All 13 downstream issues have status, remaining work, and full-workload completion
criteria above their preserved historical reports. Later
[PR #114](https://github.com/Lulzx/cuda-metal/pull/114) reports clearing Bitcoin
self-test pointer-field errors and reaching the trap-expansion limit. Subsequent
targeted local research on `92a9b8f4de23` confirms that it **does not clear the
RSA-PSS self-test's 63 pointer errors**. The missing mixed-vector case now has
its own [#118](https://github.com/Lulzx/cuda-metal/issues/118). A diagnostic copy
with that single vector reload split into scalar loads clears those errors and
reaches a trap-expansion limit; no complete self-test pass is established.

Newly published [PR #117](https://github.com/Lulzx/cuda-metal/pull/117) proposes
preserving guarded helper calls for #116. It reports full PTX translation for
Bitcoin, Solana, Ethereum, and P-256 public-key self-tests, with Bitcoin still
waiting in Metal compilation after ten minutes. Those separately reported
results are outside both this complete run and the targeted `92a9b8` research.
See the [ownership matrix](cumetal-issue-matrix.md) for current scope and evidence.

## What the production checks cover

Every production command requested
`--blocks 1 --threads-per-block 32 --batches 2 --seed 1 --verify`.
The address/public-key patterns were empty; Shallenge started with an all-ones
target. Signature inputs were the retained disposable PKCS#8 P-256 and RSA-2048
keys and 37-byte message from the historical run. Message search used an
eight-byte nonce window at offset eight; RSA-PSS message search used a fixed
32-byte zero salt. Crypto search preparation can also use OS randomness.

For the two passing modes, `--verify` evaluated all 64 candidates on the CPU,
compared aggregate match/error counts, and compared each returned winner's bytes.
The host independently verified exported results. The transport checked unchanged
inputs and buffer guards. The compact GPU result contains one winner per batch,
so this does not compare every candidate's output bytes. Two batches exercise
buffer reuse. These are bounded smoke checks, not exhaustive pattern,
launch-geometry, or performance validation.

Each production command and each self-test group had a **300-second wall-clock
limit**, including translation, Metal compilation, execution, and verification.
Commands ran sequentially. On timeout the harness terminated only its own client
process group; system compiler services can outlive a client and affect subsequent
timings. No unrelated process was terminated.

## Tested artifacts

- Hardware: Apple M5; macOS 26.6.2.
- Host source: `55ae44bcd1c1bfa116712212cac22a9598497fbe`, with the working-tree
  `flake.nix`, `flake.lock`, and README changes captured in an isolated snapshot.
  Cargo built into a separate target directory, then the executable was copied
  into the evidence bundle before testing.
- PTX producer source: `a2aba42c3dbaa876a25cdc3ddd99c2c7ec58d3bb`.
  All 92 audited kernel/shared-logic/build-input files match that producer, and
  the GPU build function is unchanged. All 16 retained PTX hashes match both
  historical manifests. The PTX was reused, not regenerated for this run.
- Original PTX producer directory:
  `vanity-nixos:/tmp/vanity-retirement-target/release/ptx/`;
  LLVM 21.1.8, CUDA 13.3, PTX 9.3, `sm_100`.
- CuMetal source: `https://github.com/brandonros/cuda-metal`, commit
  `e5acf8cc0c658142c704ee80e749f5180911fff4`.
- Source tree: `/nix/store/r7mh28k40g28rl7qf0gi7nnnfk3wcrj0-source`.
- Paired compiler/runtime package:
  `/nix/store/39dqnk97w6kbspvh26jha6n90q1rkg72-vanity-cumetal-e5acf8cc0c65`.
  Compiler: `bin/cumetalc`; runtime: `lib/libcumetal.dylib`.
- Compiler SHA-256:
  `edefe1339a26f9b657acd1f1dc08aa1f6f6c13e2bcb8d89cf907d87c0e844f38`.
- Runtime SHA-256:
  `09504f3f3332714e777abdbce60dca1b892c47f998532432c9e13afaccd9577d`.
- Host executable SHA-256:
  `d1f40ca7dc08d88d570061eb5c1b757792cd26e415f2ce02879675e9c7b48a2b`.

Each command translated its recorded PTX snapshot using
`--backend=cumetal-ir --ptx-strict --emit=msl`. Local contribution worktrees,
alternate compiler/runtime binaries, and precompiled Metal inputs did not supply
these results.

## Commands and local evidence

The isolated source build succeeded:

```sh
# From the frozen source/ directory in the evidence bundle:
nix develop path:.#cumetal --command cargo build --offline --release --locked \
  --target-dir ../target -p vanity-miner --no-default-features \
  --features cumetal,self_test,solana,bitcoin,ethereum,shallenge,rsa-modulus,rsa-pss,p256-public-key,p256-signature
```

The harness passed the absolute paired package path with `--cumetal-root` and the
bundle's PTX directory with `--ptx`. `results.json` records all 18 exact commands,
exit codes, elapsed times, limits, log paths, and hashes of retained timeout
artifacts. `audit.py` verified coverage of all 16 PTX inputs and 160 selected
self-test slots, input/source hashes, and the compiler/runtime identity in every
log.

Local evidence directory (ignored by Git):
`.cumetal-artifacts/validation-20260915T183226Z/`.

- `metadata.json`, `run-context.json`: machine, source/build identities, timing,
  build command, and background compiler context.
- `source/`, `source-working-tree.diff`, `vanity-miner`, `logs/host-build.log`:
  isolated source, recorded working changes, exact executable, and successful build.
- `input-hashes.json`, `producer-sha256.txt`, `ptx-provenance.json`, `ptx/`:
  tested inputs and source compatibility evidence.
- `run.py`, `results.json`, `logs/`: bounded harness and complete run evidence.
- `self-test-inventory.json`, `parse_results.py`, `self-test-results.json`:
  ownership, parsing rules, and observed outcomes for all 160 slots.
- `audit.py`, `audit.json`: completeness and identity checks.
- `fix-coverage-audit.json`: included PR commits, exact source comparisons,
  implementation gaps, and limits of the inspected PR claims.
- `temporary/`: retained Metal sources and ABI sidecars from timed-out commands;
  their hashes are in `results.json`.
- `logs/*-process-sample.txt`: live wait-state evidence for all three timeout commands.
- `historical/`: previous Markdown reports, preserved before replacement.
- `fixtures/`: disposable test inputs. Full production logs can contain generated
  test private keys; they remain local validation artifacts.

The complete historical run is retained separately under
`.cumetal-artifacts/validation-20260915T151954Z/`. Its results describe `98cf505`,
not the newer tested revision. Earlier host tests are also separate evidence;
this rerun rebuilt the host and exercised the actual CuMetal backend.

## PTX SHA-256 manifest

Paths are relative to the local evidence directory above.

| Input | SHA-256 |
| --- | --- |
| `ptx/bitcoin.ptx` | `9a8a5a73be2df6c18934894f7529ff028dca773dfd987f9f7eae53cd61ee0d86` |
| `ptx/ethereum.ptx` | `12b02ff8e0046d8909b99c881de0ea80c48e6716297f62a3cddf27b18870f0fa` |
| `ptx/p256_public_key.ptx` | `18c9562f16d1010194589e56becea53808ec99f2d7ad8f0510cdd3cd7e6c9812` |
| `ptx/p256_signature.ptx` | `99ad3d9f77cefc08dfa61e39d3530013aa7a8e0bc0b35c06775bfb14c0b6efd2` |
| `ptx/rsa_modulus.ptx` | `13392fd770156ac337929724d0d5d8ca3f4b17314c0455e6bb8668d7648a9b1a` |
| `ptx/rsa_pss.ptx` | `ab2b4afb1d9c4d57f165ccef7297a753ac815d126d886534ae95a3330d1018ad` |
| `ptx/self_test_bitcoin.ptx` | `1f16b59a3719a862d7aa56636f5392ca45837d1ecb1bd5778e31554d766d18a8` |
| `ptx/self_test_ethereum.ptx` | `b52aeb7e75ef61a25166658644c6cfbc9fc6ef1e7454176677d745503617872d` |
| `ptx/self_test_p256_public_key.ptx` | `a83bdc53edb56127a79bbbe0f638ef7d3a0d9125c419a775e40477e8b8b5ec19` |
| `ptx/self_test_p256_signature.ptx` | `8e5d9aba67f58a75ad2e22afd46e7b7bdd8324bd963a6628e79e62ea5edc0235` |
| `ptx/self_test_rsa_modulus.ptx` | `b0125074920946a820c9c0bd84daf5493b8fe8578a563593522219f71fccbe7b` |
| `ptx/self_test_rsa_pss.ptx` | `29e163809de5a5027390ad10d6c0f2d7ff75bf87cbaccaafee7f20380009ef33` |
| `ptx/self_test_shallenge.ptx` | `3dcbe7abd749dc1e87e2150f89e471c4ea914332f85dd55ba5081775861414b7` |
| `ptx/self_test_solana.ptx` | `e9ef053adf5b4f70d3607dcbae3890ae5886cfe59303b8f436be012ee47bf9e2` |
| `ptx/shallenge.ptx` | `e39509140aaf2cb8272729c2888790788d6c24edcd2b65489fa19d55b1a042c3` |
| `ptx/solana.ptx` | `cae885f153fe426c2d99672301bb1cc7ec5e7506a77ab8481297c14d0a45e0db` |
