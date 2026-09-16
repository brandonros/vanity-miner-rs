# CuMetal validation — 2026-09-16

RSA source update: production now uses one resumable `kernel_rsa_modulus_vanity`
entry. The RSA GPU measurements below concern the historical four-stage PTX,
not this refactor; the updated RSA self-test also needs fresh GPU validation.
See [the mining design](gpu-search-pipeline.md).

## Result

Fresh measurement **2026-09-16, 01:46:07–02:14:38 UTC**:
**2 of 16 rows pass; 14 fail**. The passing rows are **Shallenge production** and
**P-256 public-key production**. See the [16-row report](cumetal-status.md).

The failures are tracked by **14 open workload issues (#23–35 plus #37)**.
Adding [#37 — Shallenge self-tests](https://github.com/brandonros/vanity-miner-rs/issues/37)
updates tracking only; no new measurements were made.

- **Production:** 2 modes pass, 4 fail PTX translation, 2 time out. Both search
  sources of each signature mode were attempted, giving 10 production commands.
- **GPU self-tests:** all 8 groups attempted; **203 checks blocked before execution**,
  0 passes, 0 numerical failures, 0 skips, 0 missing outcomes. Six modules fail PTX
  translation; two emit Metal that Apple rejects.
- **CPU baseline:** the matching source passes all **203 checks**, with 0 failures
  or skips. This establishes the CPU fixtures, not GPU correctness.

The final evidence audit passes: all **18 commands**, **16 PTX inputs** and
**203 named self-test selections** are accounted for, and source/input/package
identities match. Audit success means the report is complete and internally
consistent; it does not turn failed workload rows into passes.

## Production results

Times below are whole-command wall times, including translation, preparation,
execution and verification. The limit was 300 seconds, plus process termination.

| Mode | Result | Seconds | Fresh evidence |
| --- | --- | ---: | --- |
| Solana | PTX translation failed | 5.44 | PTX line 34070: `i32` operand refers to an `i64` value. |
| Bitcoin | PTX translation failed | 11.22 | Line 26556: pointer subtraction requires a pointer minus a 64-bit integer byte offset. |
| Ethereum | PTX translation failed | 10.52 | Same diagnostic, line 22069. |
| Shallenge | **Passed** | 2.03 | Two batches, 64 candidates, two returned outputs; CPU verification and buffer checks pass. |
| P-256 public key | **Passed** | 192.36 | Two batches, 64 candidates, two returned outputs; CPU verification and buffer checks pass. Samples at 60/180 seconds showed Metal pipeline creation before eventual success. |
| P-256 signature: message | PTX translation failed | 35.55 | `%rd17653` undefined on an incoming edge to `$L__BB0_11`. |
| P-256 signature: ephemeral | PTX translation failed | 38.14 | Same diagnostic. |
| RSA modulus | Timed out | 301.22 | All four Metal sources emitted. The 180-second and late samples showed `newComputePipelineStateWithFunction`, not a completed GPU cycle. |
| RSA-PSS: salt | Timed out | 301.67 | Emitted 24,263,920-byte Metal source; 180-second and late samples showed `newLibraryWithSource`. No completed candidates. |
| RSA-PSS: message | Timed out | 301.21 | Same emitted source size; independently sampled in `newLibraryWithSource`. No completed candidates. |

The measured eight production modules had eleven entry points because that RSA
modulus artifact used four stages. Rebuilt sources now have eight production entries. Translating all four does not establish execution of all four.
In particular, this fresh RSA modulus observation is a **pipeline-creation wait**;
the historical sample waiting for GPU command completion is separate evidence.

## Self-test results

Each group ran in its own process with `self-test --check MODE.CHECK`, selecting
every owned name exactly once. Selection runs the containing kernel in full.
A module preparation error is reported against every selected check in that
module; **203 blocked checks are eight blocked modules, not 203 numerical bugs**.

| Group | Checks blocked | Stage | First observed diagnostic / location | Seconds |
| --- | ---: | --- | --- | ---: |
| Solana | 79 | PTX translation | `%rd42` / `$L__BB77_6`, new `solana.candidate_match`. | 77.45 |
| Bitcoin | 39 | PTX translation | `%r1688` / `$L__BB4_3`, `bitcoin.private_key`; undefined high half packed before a narrower use. | 59.96 |
| Ethereum | 8 | PTX translation | Pointer-subtraction diagnostic at PTX 17373, new `ethereum.candidate_match`. | 21.22 |
| Shallenge | 21 | PTX translation | `%rd16` / `$L__BB18_1`, new `shallenge.sha256_streaming_chunks`. | 15.15 |
| P-256 public key | 13 | Apple Metal compilation | Undeclared `private$39` at MSL 92833; additional pointer address-space errors. | 26.54 |
| P-256 signature | 13 | PTX translation | `%p243` / `$L__BB8_1`, `p256_signature.low_s`. | 50.02 |
| RSA modulus | 16 | PTX translation | `%rs907` / `$L__BB16_1`, new `rsa_modulus.range_multiple`. | 95.21 |
| RSA-PSS | 14 | Apple Metal compilation | MSL 138317: illegal `as_type` from `ulong` to `device uchar*`; later bracketed global expressions cause syntax errors. | 165.95 |
| **Total** | **203** | **Before GPU execution** | **Every selected check reported once; no numerical results or skips.** | |

`rsa_pss.end_to_end` is the sole intentional GPU skip (current slot 183). Because
its module failed first, the skip was not reached. A successful suite under this
policy would normally report **202 GPU passes and one skip**.

The additional Metal errors matter independently:

- **P-256 public key:** emitted helpers reference undeclared `private$39`; another
  expression casts through `cm_alias_uchar*` without an address-space qualifier.
  Fixing the missing name alone is not evidence that every diagnostic is fixed.
- **RSA-PSS:** besides the invalid pointer cast, emitted expressions such as
  `ulong([private$em])` retain PTX-style brackets. Apple parses these as unsupported
  lambda syntax. The PTX global is used by the CRT known-answer/fault/modulus
  helpers. The log contains 1 invalid-cast, 96 lambda and 192 parenthesis errors;
  those counts do not represent independent root causes.

The CLI cleaned up generated Metal after these two failed commands. PTX, compiler
identity and full diagnostic logs remain in the bundle. Some MSL locations cannot
be mapped uniquely back to a helper without regenerating the source. No speculative
root-cause assignment is presented as a completed reduction.

## What changed, and what has not been proved

The historical full run used CuMetal `e5acf8cc0c65`, producer `a2aba42`, and 160
checks; it had **3 true / 13 false**. This run uses `9e3e615`, producer `4e0231a`,
and 203 checks. It is **not a pin-only comparison**.

- **Shallenge self-tests add the fourteenth failing row.** The old eight-check
  module passed; the expanded 21-check module fails in a newly added streaming
  check. [Downstream #37](https://github.com/brandonros/vanity-miner-rs/issues/37)
  now tracks this new row; its upstream owner remains unconfirmed.
- **All eight production PTX modules have identical instructions, registers,
  control-flow labels and data after an explicit, consistent renaming of declared
  module symbols/parameters.** All original byte hashes differ. The comparison
  excludes PTX instruction growth as the explanation of the production timing
  differences, but does not prove identical generated Metal or cache behavior.
- **The rewritten self-test inputs expose different first blockers.** Included
  PRs #117, #121 and #122 retain their demonstrated scope on the original
  reproductions. They do not establish a pass for these changed modules.
- **RSA-PSS self-tests now emit Metal.** This changed input getting past PTX
  translation does not establish that the unimplemented mixed-vector correction
  in upstream #118 was added or that its original reproduction is fixed.
- **RSA modulus self-tests stop earlier than the historical Metal allocation
  failure**, in the new range check. Upstream #124/#127 resource work remains
  historical follow-up work, not the observed first stage of this fresh input.

The [issue matrix](cumetal-issue-matrix.md) separates original upstream owners,
included scoped fixes, fresh observations, and cases needing further reduction.
Similar register/type diagnostics alone do not establish duplicate defects.

## Scope and limits of the checks

Production commands used `--blocks 1 --threads-per-block 32 --batches 2 --seed 1
--verify`. Address/key patterns were empty; Shallenge used an all-ones target.
Both signature search sources used retained disposable keys and the 38-byte
message. Message search used an eight-byte nonce window at offset eight;
RSA-PSS message search used a fixed 32-byte zero salt. Some crypto preparation
uses OS randomness, so `--seed 1` does not freeze every candidate.

For the two passing modes, verification evaluates all 64 candidates on the CPU,
compares aggregate match/error counts and each returned winner's bytes, checks
unchanged inputs and buffer guards, and verifies exported results. The compact
result returns one winner per batch; it does not compare all candidate output
bytes. Two batches exercise buffer reuse. These are bounded correctness checks,
not exhaustive pattern, launch-geometry or performance validation.

Every command had a **300-second whole-process limit** and ran sequentially.
Only its own process group was terminated on timeout. Shared Metal caches were
not cleared and compiler services can outlive a client. Elapsed times are not
controlled benchmarks. Stack samples identify observed call sites, not exact
per-stage durations or root causes. The timing diagnostics in PR #128 are
**not in this pin**. Module-load success alone is never counted as compilation
or GPU success.

## Source and artifact identity

- Hardware: **Apple M5**, macOS **26.6.2**, build `25G83`.
- Host and PTX producer: **`4e0231aa82b69be146936f53829c95ef3f522832`**.
  The host was built from a frozen Git archive of that exact Actions commit.
  Later working-tree edits were not used.
- PTX: [successful Actions run #35044328837](https://github.com/brandonros/vanity-miner-rs/actions/runs/35044328837),
  [LLVM 21 job](https://github.com/brandonros/vanity-miner-rs/actions/runs/35044328837/job/104630788363),
  artifact **`ptx-llvm21` / `10426657549`**. LLVM **21.1.8**, CUDA **13.3**, PTX
  **9.3**, target **`sm_100`**. No Cargo cache was restored; producer compilation
  completed in 11m 28s. The uploaded archive digest was verified before extraction.
- LLVM 7 also built successfully; its companion artifact is retained but **was
  not used for these CuMetal results**.
- Rust-CUDA revision: `f554f74a78a1ee30b2668f0ec2e5b1fd54b7b588`;
  Rust toolchain: `nightly-2026-04-02`.
- CuMetal: **`9e3e61574b776424a96c686bdbdc04ad1f27fe9f`**, from
  `https://github.com/brandonros/cuda-metal`, matching the selected `flake.lock`.
- Paired package: `/nix/store/hh6l39zigklrh3al2w21x7b8ik5jaxcm-vanity-cumetal-9e3e61574b77`.
- Immutable CuMetal source: `/nix/store/kk52zigih0j1x7dik3fbf9jwjl1ka139-source`.

| Artifact | SHA-256 |
| --- | --- |
| `bin/cumetalc` | `5662a763e6e63539cd9776efb844187c707d23297c35d2950c1e5820eb9ea628` |
| `lib/libcumetal.dylib` | `2b270e4154b9c16df64ee75f33655a37d7fea843a61055ff180ce3bd515ba549` |
| CuMetal host executable | `19b4c93129a13841d7434d1c3c6d3616300d88ae41851a7eeb2b2885ecf05072` |
| Uploaded LLVM 21 artifact ZIP | `a7d8f2ff1f7892cf73f442b7bce5f33f54eb6db7c718105998449417d3d6d144` |
| Inner LLVM 21 PTX archive | `c2188938132bf7dece2d1704ec48855dfc10a72e8de4db5275d19aedef9e7bc0` |

Every command verified and printed the paired package identity and its PTX
hash. Local CuMetal contribution worktrees and the cancelled preliminary Lima
build supplied none of the measured PTX/compiler/runtime inputs.

## Reproduction and retained evidence

From the frozen `source/` directory:

```sh
nix develop path:.#cumetal --command cargo build --offline --release --locked \
  --target-dir ../target -p vanity-miner --no-default-features \
  --features cumetal,self_test,solana,bitcoin,ethereum,shallenge,rsa-modulus,rsa-pss,p256-public-key,p256-signature
```

`run.py` records each exact command, using the absolute paired `--cumetal-root`
and extracted `--ptx` directory. `audit.py` verifies complete coverage and hashes.
Use a new bundle for a rerun; the harness refuses to overwrite existing results.

Local evidence (ignored by Git):
`.cumetal-artifacts/validation-actions-35044328837-20260916T013211Z/`.

- `metadata.json`, `source/`, `run-context.json`, `vanity-miner`: exact consumer,
  source hashes, machine/measurement context and binary.
- `github-run.json`, `github-artifacts.json`, `ptx-provenance.json`, `downloads/`,
  `ptx/`, `logs/actions-llvm21.log`: producer evidence, verified archives and inputs.
- `input-hashes.json`, `self-test-inventory*.json`, `cpu-baseline.json`: fixtures,
  current registry, host inventory cross-check and 203-pass CPU baseline.
- `run.py`, `results.json`, `logs/`, `temporary/`: 18 exact commands, outcomes,
  diagnostic/stack logs and retained timeout artifacts.
- `audit.py`, `audit.json`, `self-test-results.json`: successful final coverage and
  identity audit, with every blocked check identified.
- `ptx-structural-comparison.{json,md}`, `fresh-self-test-triage.json`: exact
  production symbol mappings and qualified locations of fresh self-test blockers.

Historical evidence remains in `.cumetal-artifacts/validation-20260915T183226Z/`;
its older report is preserved in Git history. No historical result is substituted
for a fresh result above.
