# CuMetal issue ownership and fix status

GitHub and commit ancestry audited **2026-09-16, 01:24 UTC**.
All 13 downstream issues below remain open. This is a map of implementation
scope and evidence; workload results belong in the [16-row status report](cumetal-status.md)
and [validation report](cumetal-validation.md).

## Current pin and validation boundary

- **Consumer pin:** `9e3e61574b776424a96c686bdbdc04ad1f27fe9f` in `flake.lock`
  ([PR #122](https://github.com/Lulzx/cuda-metal/pull/122)). Git ancestry confirms
  it includes #114, #117, #121 and their prerequisite stack.
- **Not in this pin:** [tracing PR #128](https://github.com/Lulzx/cuda-metal/pull/128),
  head `6b70a1b618c4ab1d80e094991941aa56970aff39`. It is a standalone change on
  upstream `main`, not a descendant included by #122. Uncommitted #123 work is
  also outside the locked source.
- **Last complete measurement, historical:** `e5acf8cc0c65` on 2026-09-15,
  18:36–18:59 UTC: **3 true / 13 false**. Later targeted translations and
  compiler-resource investigations do not replace that complete run.
- **Fresh measurement at `9e3e615`: awaiting Actions PTX.** The selected
  [kernel build](https://github.com/brandonros/vanity-miner-rs/actions/runs/35044328837)
  produces PTX from `4e0231aa82b69be146936f53829c95ef3f522832`. Its matching
  CuMetal host is built; all **203 CPU self-tests pass**. GPU results are pending.
  The rewritten registry has 203 checks, including one intentional GPU skip;
  historical group/slot counts and PTX must remain separately identified.

## Read the statuses separately

| Status | What it establishes |
| --- | --- |
| **No published fix PR** | The live issue/PR audit found no patch for that remaining case. It does not establish that nobody has started locally. |
| **Local WIP; no PR** | An explicitly reported unpublished implementation exists. It is outside the pin unless committed and selected. |
| **PR proposed / included** | Code exists in a published PR; the separate inclusion column says whether this build contains it. Neither means the full workload passes. |
| **Blocker cleared; next failure remains** | An identified artifact passes the fixed stage and reaches another failure. Preserve the first fix's scope and name the next owner. |
| **Workload verified** | The identified workload completes GPU execution and its required result/guard/CPU checks. Emitted Metal or successful module-handle creation is insufficient. |

All PRs named as current fixes below are open drafts at this audit. “Included”
means present in the locked contribution stack, not merged into upstream `main`.
Do not label a workload “solved” merely because its original error disappeared.

## Downstream-to-upstream matrix

The evidence column summarizes **previously recorded, identified artifacts**;
it is not the pending fresh measurement. Completion requires each current
production mode's verification or every enabled self-test belonging to its group,
with intentional skips reported separately.

| Vanity-miner issue | Upstream owner(s) | Published implementation | Included in `9e3e615`? | Latest evidence and remaining work |
| --- | --- | --- | --- | --- |
| [#23 — Solana production](https://github.com/brandonros/vanity-miner-rs/issues/23) | [#76 — conversion/join types](https://github.com/Lulzx/cuda-metal/issues/76) | **No PR for the remaining destination/normalized-definition/join work.** #50/#56/#75 cover narrower cases. | Earlier PRs only. | Historical `e5acf8` type failure. Implement destination inference before SSA and preserve definition/join types; then translate and CPU-verify batches. |
| [#24 — Bitcoin production](https://github.com/brandonros/vanity-miner-rs/issues/24) | [#76](https://github.com/Lulzx/cuda-metal/issues/76) | **No PR for this remaining gap.** Shared implementation with #23/#25. | Earlier PRs only. | Historical widened-offset/pointer-subtraction failure. Retest this artifact separately after the shared type correction. |
| [#25 — Ethereum production](https://github.com/brandonros/vanity-miner-rs/issues/25) | [#76](https://github.com/Lulzx/cuda-metal/issues/76) | **No PR for this remaining gap.** Shared implementation with #23/#24. | Earlier PRs only. | Same historical diagnostic class as Bitcoin; needs its own verified GPU batches. |
| [#26 — P-256 signature production](https://github.com/brandonros/vanity-miner-rs/issues/26) | [#83 — empty labels/predicate facts](https://github.com/Lulzx/cuda-metal/issues/83) | #85 implements original predicate cases; **no empty-label follow-up PR**. | #85 and #121 included; #121 changes a different depth case. | Historical consecutive-label failure. Preserve facts through empty blocks and validate both message and ephemeral searches; do not count #121 as this fix. |
| [#27 — RSA modulus production](https://github.com/brandonros/vanity-miner-rs/issues/27) | [#119 — generation completion](https://github.com/Lulzx/cuda-metal/issues/119) | **No fix PR.** | No completion fix. | Historical host generation launch waits during device synchronization after pipeline creation. Actual pending-command identity/cause is unproved. Diagnose it and complete verified four-stage cycles. |
| [#28 — RSA-PSS production](https://github.com/brandonros/vanity-miner-rs/issues/28) | Earlier #96/#109/#111 → [#115 — Metal compilation time](https://github.com/Lulzx/cuda-metal/issues/115) | #108/#110/#112 clear translation; [#128](https://github.com/Lulzx/cuda-metal/pull/128) adds **diagnostics only**. **No performance fix PR.** | Translation fixes yes; #128 **no**. | Historical full PTX emits MSL; salt/message searches time out while sampled in Metal compilation. Establish and correct the resource bottleneck, then verify both searches. |
| [#29 — Solana self-tests](https://github.com/brandonros/vanity-miner-rs/issues/29) | [#46](https://github.com/Lulzx/cuda-metal/issues/46) / [#116 — trap helpers](https://github.com/Lulzx/cuda-metal/issues/116) | [#117](https://github.com/Lulzx/cuda-metal/pull/117) preserves trap-capable helper calls. | **Yes.** | PR #117 reports retained full-module translation. Current group preparation/execution/results still require the fresh run. |
| [#30 — Bitcoin self-tests](https://github.com/brandonros/vanity-miner-rs/issues/30) | [#113 — pointer fields](https://github.com/Lulzx/cuda-metal/issues/113) → [#116](https://github.com/Lulzx/cuda-metal/issues/116); later compile wait **unassigned** | [#114](https://github.com/Lulzx/cuda-metal/pull/114), then [#117](https://github.com/Lulzx/cuda-metal/pull/117). **No correction for the subsequent wait.** | **Yes**, both scoped fixes. | #117 reports translation, followed by >10 minutes in Metal compilation. Compare with #115/#124 without assuming the same cause; assign a demonstrated later defect and execute the group. |
| [#31 — Ethereum self-tests](https://github.com/brandonros/vanity-miner-rs/issues/31) | [#46](https://github.com/Lulzx/cuda-metal/issues/46) / [#116](https://github.com/Lulzx/cuda-metal/issues/116) | [#117](https://github.com/Lulzx/cuda-metal/pull/117). | **Yes.** | Retained full-module translation reported by #117; current complete GPU validation pending. |
| [#32 — P-256 public-key self-tests](https://github.com/brandonros/vanity-miner-rs/issues/32) | [#46](https://github.com/Lulzx/cuda-metal/issues/46) / [#116](https://github.com/Lulzx/cuda-metal/issues/116) | [#117](https://github.com/Lulzx/cuda-metal/pull/117). | **Yes.** | Retained full-module translation reported by #117; current complete GPU validation pending. |
| [#33 — P-256 signature self-tests](https://github.com/brandonros/vanity-miner-rs/issues/33) | [#120 — guard depth](https://github.com/Lulzx/cuda-metal/issues/120), within [#83](https://github.com/Lulzx/cuda-metal/issues/83)'s broader scope; later types **unassigned** | [#121](https://github.com/Lulzx/cuda-metal/pull/121) removes the redundant eight-block cap. | **Yes.** | #121 reports `%rs506` cleared, then independent type-provenance diagnostics. Capture/reduce the next error before assigning it to #76/#118 or another issue. No full numerical pass. |
| [#34 — RSA modulus self-tests](https://github.com/brandonros/vanity-miner-rs/issues/34) | [cuda-metal #35 — masked payloads](https://github.com/Lulzx/cuda-metal/issues/35) → [#124 — Metal allocation failure](https://github.com/Lulzx/cuda-metal/issues/124); proposed optimization [#127](https://github.com/Lulzx/cuda-metal/issues/127). Changed outlined input: [#123](https://github.com/Lulzx/cuda-metal/issues/123)/[#125](https://github.com/Lulzx/cuda-metal/issues/125). | [#122](https://github.com/Lulzx/cuda-metal/pull/122) clears masked-payload SSA. **No resource/PRMT fix PR.** #123 is unpublished local WIP. | #122 **yes**; later work **no**. | Original PTX translates to 53.6 MB MSL, then Apple compilation aborts. Outlining changes PTX and exposes helper-global defects; it is not an original-input fix. Measure #127 and verify the current group separately. |
| [#35 — RSA-PSS self-tests](https://github.com/brandonros/vanity-miner-rs/issues/35) | [#118 — mixed local-vector pointer lanes](https://github.com/Lulzx/cuda-metal/issues/118); subsequent trap work related to [#116](https://github.com/Lulzx/cuda-metal/issues/116) | **No PR for #118.** #108/#114 cover other pointer cases; #117 covers trap helpers. | Related PRs yes; mixed-lane correction **no**. | Historical unchanged PTX has 63 pointer diagnostics. A modified scalar-load probe clears them and exposes expansion; unchanged input plus combined fixes and complete enabled-check validation remain required. |

## Additional upstream work and boundaries

| Upstream issue | State at audit | Why it matters / what remains |
| --- | --- | --- |
| [#123 — helper global arguments](https://github.com/Lulzx/cuda-metal/issues/123) | **Local WIP; no commit/PR.** Public handoff records code and focused tests outside the pin. | Changed, outlined RSA input needs globals threaded through helper signatures/calls. Exact source translation clears the symbol error, but later Apple allocation aborts remain; full baseline gates and publication are pending. |
| [#125 — helper-only registration globals](https://github.com/Lulzx/cuda-metal/issues/125) | **No fix PR.** | Separate metadata traversal defect: reachable helper-only globals miss initialization/persistence. The #123 small GPU fixture retains a direct entry reference to bypass this gap. |
| [#124 — Metal allocation failure](https://github.com/Lulzx/cuda-metal/issues/124) | **Research; no demonstrated fix.** | Original monolithic and changed outlined artifacts must retain separate results. A successful `cuModuleLoad` does not prove either source compiled. |
| [#127 — constant byte-shuffle specialization](https://github.com/Lulzx/cuda-metal/issues/127) | **Optimization not implemented; no PR.** | First concrete experiment for #124: immediate `prmt.b32` currently creates 66 values per shuffle. Its sequences occupy 35.55% of the retained RSA entry's source bytes; neither speedup nor crash correction is established. |
| [#115 / tracing PR #128](https://github.com/Lulzx/cuda-metal/pull/128) | **Diagnostic implementation proposed; outside pin.** Performance remains unresolved. | Adds opt-in compilation spans; an end record means scope exit, including failure. New tests pass, but wider Release/selected Debug failures are documented. It neither supplies #126 nor fixes #124. |
| [#126 — explicit function preparation](https://github.com/Lulzx/cuda-metal/issues/126) | **Missing APIs; no fix PR.** | Add `cuFuncLoad`/`cuFuncIsLoaded` with shared preparation, readiness, errors and lifecycle tests. Preparation success remains separate from GPU results and hidden-buffer correctness. |

## Shared fixes and next assignments

- **One type-inference owner:** downstream #23–25 share #76. Coordinate its
  importer result-type edits with #118's per-lane pointer work.
- **One CFG owner:** #121 and #122 are separate implemented fixes, not duplicate
  repairs. Remaining #83 empty-label work must preserve both sets of regressions.
- **One helper-call owner:** #117 serves #29–32; #123/#125 are subsequent
  compiler/registration work on helpers with globals.
- **Separate resource investigations:** #115 (RSA-PSS compilation), #124 (RSA
  self-test allocation failure), and #119 (production command completion) have
  different observed stages. Similar symptoms alone do not establish duplicates.
- **One integration owner:** control the pin, pair compiler/runtime artifacts,
  preserve PTX identities and serialize GPU measurements. A fresh result should
  name the next failure stage without rewriting a scoped fix as a workload pass.

The research below preserves earlier evidence. It is not a new execution result
on the current source or a current slot inventory.

## Historical research: RSA modulus production (#27)

[Upstream #119](https://github.com/Lulzx/cuda-metal/issues/119) owns the execution
investigation. Disassembly of the exact retained host executable identifies
`RsaPipeline::cycle +184` as the first `Module::launch`, for `kernel_rsa_generate`.
The live sample is waiting for Metal command completion after pipeline creation.
It does not distinguish queued work, execution, or a GPU-side dependency. The
300-second limit measures the whole process, not GPU execution alone.

The production RSA pipeline gets its candidate seed from `OsRng`; top-level
`--seed 1` did not make these candidates reproducible. A deliberately public
zero-seed fixture with the default interval was used for bounded CPU diagnostics:

| Diagnostic | Result | Limit of the evidence |
| --- | --- | --- |
| Generated MSL adapted to scalar C++, 32 sequential lanes | 1.130 s; 47,516,176 dispatcher steps; all candidates rejected; no traps or reported errors. | Address spaces were erased and `simd_any` reduced to scalar identity. This does not validate Metal or SIMD execution. |
| Native Rust logic, same public fixture | 18.130 ms including classification/logging; matching rejection statuses; nine candidates entered Miller–Rabin. | Intermediate candidate bytes/arithmetic were not compared, and no accepted-prime fixture was tested. |

These timings are diagnostic observations, not GPU performance estimates. The
scalar/native status agreement does not establish a correct GPU translation.
A separate static omission in Driver API hidden-global binding was found, but
none of the 32 scalar fixture lanes read that global; its connection to the
observed wait is unproved. It remains a qualified lead within #119.

The next steps are attributable command-phase/error records, fixed public GPU
fixtures covering rejection and expensive primality paths, CPU comparison, and
a reduced demonstration of the actual fault before choosing a correction.
Completion still requires the original four-stage pipeline and verified cycles.

- GPU evidence: complete `e5acf8cc0c65` run; no additional GPU run was performed.
- PTX SHA-256: `13392fd770156ac337929724d0d5d8ca3f4b17314c0455e6bb8668d7648a9b1a`.
- Generation MSL SHA-256: `5de9a0b3bf2a349523f5ff2bb5695cb58134f9fb06f23821514841f83caad1e8`.
- Local evidence: `.cumetal-artifacts/ownership-research-20260915/issue-27/`.

## Historical research: RSA-PSS self-tests (#35)

On **2026-09-15**, the unchanged full PTX was translated with the immutable
`92a9b8f4de23` compiler. It failed with the same **63 diagnostics** as `e5acf8cc0c65`,
starting at line 2377. The seed pointer and integer length are reloaded together
by `ld.local.v2.b64`; the importer incorrectly requires one pointer type across
both lanes. PR #114 handles a different, scalar helper-load case.

Eight small compiler probes confirmed that both mixed-vector layouts fail and
scalar/direct controls emit MSL on both revisions. Splitting only that reload
in a diagnostic copy of the full PTX clears all 63 errors and reaches:

```text
trap call expansion exceeds bounded CFG size (calls=1, blocks=11836, operations=649614)
```

This is strong evidence for the first defect and evidence of a subsequent
blocker. The changed-input diagnostic is not a compiler fix or a GPU pass.
PR #117 is now included in the consumer pin, but must still be tested against
this RSA-PSS case after pointer recovery is corrected. [Upstream #118](https://github.com/Lulzx/cuda-metal/issues/118)
contains the public reproducer, controlled comparisons, implementation scope,
and acceptance criteria.

- Original PTX SHA-256: `29e163809de5a5027390ad10d6c0f2d7ff75bf87cbaccaafee7f20380009ef33`.
- Diagnostic PTX SHA-256: `cab02b02abae6cda55c0c9903f27f79c2e8593427d22903a217acb4ba4313b20`.
- Compiler SHA-256: `db356e5407a3a5fcdfbea68359f02bcc7afda9bc6018803ff556c719cb8a570f`.
- Unchanged input: exit 1, 134.01 seconds. Modified input: exit 1, 148.02 seconds.
- Local evidence: `.cumetal-artifacts/ownership-research-20260915/issue-35/`.
- CPU-only compiler research; no new GPU self-test result is claimed.

See [the validation report](cumetal-validation.md) for artifact identities,
commands, failure stages, and the limits of the bounded GPU checks.
