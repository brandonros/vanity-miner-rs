# CuMetal issue matrix

Updated 2026-09-17. **Pin: `684bb0302953f0e60748a8a4ea046f9baeca37b7`
([PR #165](https://github.com/Lulzx/cuda-metal/pull/165)).** The normal all-mode
miner was rebuilt against the matched Nix compiler/runtime package. The cumulative
implementation PRs remain unmerged; #162–165 are now included. Unpublished #123
and diagnostic PR #128 remain outside this pin. The later bitwise guard follow-up
[PR #166](https://github.com/Lulzx/cuda-metal/pull/166), `21277ac`, is also outside it.

## Counts and measured progress

- **6 workload issues closed; 10 open:** three production modes and seven self-test groups.
- **32 combinations:** eight production + eight self-test workloads, each under LLVM7 and LLVM21.
  Thirteen have accepted results at the revisions below; **19 still lack complete acceptance**.
- Broad #16/#19 make **12 open vanity-miner issues total**. Shallenge self-test #37 is now closed.
- Bitcoin LLVM7 translation to the same diagnostic fell **97.076 → 22.879 → 8.120 seconds**
  after [printf](https://github.com/Lulzx/cuda-metal/pull/144) and
  [declaration-scan](https://github.com/Lulzx/cuda-metal/pull/148) fixes. No translation cache added.
  [PR #146](https://github.com/Lulzx/cuda-metal/pull/146) adds importer phase timings.
- Combined Release checkpoint: **128 passed, two skipped**, with two tool-prerequisite
  tests excluded, at development `04a0955`. The final generic-store scope narrowing
  at `0dbdb4b` passes all five affected suites; #161 passes its three focused suites
  at `70294ed`. No Debug result is claimed.
- **Shallenge self-tests pass CPU 21/21, LLVM7 GPU 21/21 and LLVM21 GPU 21/21.**
  Full CLI invocations took **34.701 s / 23.756 s**; GPU intervals were **12.488 / 13.260 ms**.
  Both versions report fresh preparation (`compile_cache_hit=false`), with all checks
  reported once and unrelated result slots intact. [Acceptance and hashes](https://github.com/brandonros/vanity-miner-rs/issues/37#issuecomment-5719149559).
- **Solana LLVM7 self-tests: CPU 79/79 and GPU 79/79 pass at `684bb03`.**
  The normal invocation completes in **330.052 s**, with fresh preparation and a
  **36.380 ms** GPU interval. The 120-second sample is in Metal pipeline creation.
  LLVM21 still rejects #140 at line 50529 in 3.780 s; #29 remains open.
  [Acceptance and hashes](https://github.com/brandonros/vanity-miner-rs/issues/29#issuecomment-5721959840).
- Shallenge production's earlier `075e963` smoke checks passed both versions:
  four GPU batches / 128 CPU-checked candidate positions. They were not rerun at this pin.

## Fix owners

“Implemented” below describes the stated compiler correction, not full workload acceptance.

| CuMetal owner | Implementation / remaining work |
| --- | --- |
| [#76](https://github.com/Lulzx/cuda-metal/issues/76) | Captured origins, bounded shifts, conditional increments and counted pointer loops in #162/#163/[#165](https://github.com/Lulzx/cuda-metal/pull/165). At `684bb03`, Solana LLVM7 passes all 79 CPU/GPU checks; Ethereum clears its private-key failure but reaches a public-key copy range at store 29927/base 816/cell 848. 26 range GPU fixtures ×65 inputs / 47 refusals pass. Historical #118 still has an unretested range-budget failure. |
| [#130](https://github.com/Lulzx/cuda-metal/issues/130) | Original #138 acceptance complete; P-256 #39 closed. Bitwise zero-marker follow-up [#166](https://github.com/Lulzx/cuda-metal/pull/166), dev `21277ac`, clears Bitcoin LLVM7 `%rs141`; later #76 pointer/scalar join remains. Five suites pass, with 22 scalar-zero GPU fixtures ×65 inputs / 74 refusals. Outside the normal pin. |
| [#134](https://github.com/Lulzx/cuda-metal/issues/134) | Implemented in #139; original packed-value blocker cleared. Later pointer proofs remain. |
| [#118](https://github.com/Lulzx/cuda-metal/issues/118) | Per-lane correction in [#145](https://github.com/Lulzx/cuda-metal/pull/145); 17 mixed-lane GPU fixtures pass in the latest group. Original full artifact now reaches a later store-range proof; current LLVM21 RSA-PSS self-tests emit MSL. Neither is a full GPU acceptance claim. |
| [#140](https://github.com/Lulzx/cuda-metal/issues/140) | Helper-field correction in #149 and compatibility follow-ups are included. Normal matched `684bb03` Solana LLVM21 still rejects an empty helper record at line 50529 (3.780 s). Context-sensitive length/pointer correlation remains; LLVM7 now passes all 79 checks. |
| [#136](https://github.com/Lulzx/cuda-metal/issues/136) | Implemented in [#153](https://github.com/Lulzx/cuda-metal/pull/153); disjoint helper writes pass GPU controls. Original Solana/Bitcoin call barriers clear; later direct-store ranges remain unresolved. |
| [#141](https://github.com/Lulzx/cuda-metal/issues/141) | CLZ in [#150](https://github.com/Lulzx/cuda-metal/pull/150); trap-capable helper support in [#159](https://github.com/Lulzx/cuda-metal/pull/159). Current LLVM21 RSA-PSS production/self-test modules emit MSL. |
| [#127](https://github.com/Lulzx/cuda-metal/issues/127) | Immediate PRMT implemented in [#154](https://github.com/Lulzx/cuda-metal/pull/154): at most 21 operations versus 66. Immediate and exhaustive runtime-selector GPU checks pass. Apple time/memory benefit unmeasured. |
| [#133](https://github.com/Lulzx/cuda-metal/issues/133) | Fresh `70294ed` LLVM21 Ethereum times out at 600.008 s in pipeline creation; CPU8/8 pass, no GPU results. A separate size-mode probe on unchanged MSL also times out at 600.012 s; Apple service sampled in LLVM function-pass execution. No performance fix or runtime option added. |
| [#151](https://github.com/Lulzx/cuda-metal/issues/151) | POPC implemented in [#156](https://github.com/Lulzx/cuda-metal/pull/156). Shared CLZ/POPC numerical tests verify 6,150 output words. RSA modulus advances to undefined incoming-register failures. |
| [#152](https://github.com/Lulzx/cuda-metal/issues/152) | Shallenge accepted at `70294ed`. #164/[#165](https://github.com/Lulzx/cuda-metal/pull/165) preserve local zero lengths across bounded writes, counted loops and `+0` calls, then simplify unsigned empty ranges. All 15 affected Release suites pass; 13 zero-guard GPU fixtures ×65 inputs / 16 refusals. Solana LLVM7 passes all 79 checks; Bitcoin/Ethereum reach later failures. #140 helper correlation stays separate. |
| [#157](https://github.com/Lulzx/cuda-metal/issues/157) | Generic-store stale cell types fixed in [#160](https://github.com/Lulzx/cuda-metal/pull/160); scalar/vector numerical and refusal checks pass. This clears LLVM7 Shallenge’s pointer/scalar join; the combined `70294ed` stack passes all21 GPU checks under both LLVM versions. PR integration pending; general #137 escape activation is separate. |

## Latest recorded workload results

**Pass means GPU execution plus CPU checks.** Revisions identify separate measurements;
untouched historical cells are not current-pin reruns. “Dev” means the development
compiler, rather than the normal paired Nix consumer. This is not a fresh 32-run sweep.

| Workload / vanity-miner issue | LLVM7 | LLVM21 | Next owner / action |
| --- | --- | --- | --- |
| Solana production [#23](https://github.com/brandonros/vanity-miner-rs/issues/23), closed | Pass `0242f22` | Pass `0242f22` | — |
| Bitcoin production [#24](https://github.com/brandonros/vanity-miner-rs/issues/24), closed | Pass `0242f22` | Pass `0242f22` | — |
| Ethereum production [#25](https://github.com/brandonros/vanity-miner-rs/issues/25), closed | Pass `0242f22` | Pass `0242f22` | — |
| Shallenge production [#38](https://github.com/brandonros/vanity-miner-rs/issues/38), closed | Pass `4a207e2`; smoke pass `075e963` | Pass `4a207e2`; smoke pass `075e963` | Self-tests are separate |
| P-256 public-key production [#39](https://github.com/brandonros/vanity-miner-rs/issues/39), closed | Pass `0f98856` | Pass `0f98856` | — |
| P-256 signature production [#26](https://github.com/brandonros/vanity-miner-rs/issues/26) | Vector parameter rejection `13efc29` | Undefined value `c4e5fac` | Reduce; #41 is a lead |
| RSA modulus production [#27](https://github.com/brandonros/vanity-miner-rs/issues/27) | Independent candidate kernel: undefined `%rs3372` into `$L__BB3_4`, matched `684bb03`; no GPU launch | Historical resumable kernel: undefined `%rd205`, dev `400a8bb`, 9.175 s | Reduce undefined edge; new ABI requires rebuilt PTX |
| RSA-PSS production [#28](https://github.com/brandonros/vanity-miner-rs/issues/28) | Vector parameter rejection `c4e5fac` | MSL emitted, dev `0dbdb4b`, 65.502 s | Apple preparation / GPU validation; historical #115 |
| Solana self-tests [#29](https://github.com/brandonros/vanity-miner-rs/issues/29) | **79/79 pass `684bb03`, 330.052 s** | Helper-field type refusal at line 50529, matched `684bb03`, 3.780 s | **#140** helper context; CPU 79/79 pass |
| Bitcoin self-tests [#30](https://github.com/brandonros/vanity-miner-rs/issues/30) | Empty suffix and masked guard clear; pointer/scalar `%rd238` join at `$L__BB20_1`, dev `21277ac`, 13.969 s | Pointer-type rejection `f7ceeef`, not rerun | #76 pointer reconstruction; #118/#152 leads for LLVM21 |
| Ethereum self-tests [#31](https://github.com/brandonros/vanity-miner-rs/issues/31) | Private-key patterns clear; public-key store-range refusal at line 30355, dev `21277ac`, 5.656 s | Pipeline timeout `70294ed`, 600.008 s; no GPU results | #76 copy range / #133 preparation cost; CPU8/8 pass |
| Shallenge self-tests [#37](https://github.com/brandonros/vanity-miner-rs/issues/37), closed | **21/21 pass `70294ed`, 34.701 s** | **21/21 pass `70294ed`, 23.756 s** | #152/#157 accepted; PRs unmerged |
| P-256 public-key self-tests [#32](https://github.com/brandonros/vanity-miner-rs/issues/32) | Undefined value `0f98856` | Metal emitted; GPU unverified `c4e5fac` | Reduce / validate |
| P-256 signature self-tests [#33](https://github.com/brandonros/vanity-miner-rs/issues/33) | Undefined value `0f98856` | Undefined value `c4e5fac` | Reduce |
| RSA modulus self-tests [#34](https://github.com/brandonros/vanity-miner-rs/issues/34) | Independent candidates: undefined `%rs3372` into `$L__BB19_4`, matched `684bb03`; all 16 checks blocked before launch | Historical resumable kernel: undefined `%rd2065`, dev `400a8bb`, 17.594 s | Reduce undefined edge; CPU 16/16 pass |
| RSA-PSS self-tests [#35](https://github.com/brandonros/vanity-miner-rs/issues/35) | Pointer/integer join rejection, dev `463c541` | MSL emitted, dev `0dbdb4b`, 93.728 s | Apple preparation / GPU checks; original #118 artifact still range-blocked |

Translation/preparation failures occur before GPU launch and do not represent
numerical assertion failures. RSA modulus LLVM7 rows record master commit `8e3e338` independent
candidate kernels (not a rerun of this Metal branch); other inputs remain producer `afe80210`,
[Actions 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
Consumer `67950b7` contains the same logic and kernel entries after the crate move.

## Next work

1. Fix #140's LLVM21 empty/mixed helper context to complete Solana's second version.
   Continue #76's Bitcoin pointer reconstruction and Ethereum correlated copy range.
   #166's Bitcoin guard correction is published but outside the pin; batch its next
   integration with the next coherent fix. Historical #118 remains a separate input.
2. Advance the now-translating LLVM21 RSA-PSS inputs through Apple preparation and numerical checks.
3. Reduce #133's costly generated functions using active compiler-service samples.
   Current PRMT output and the size-mode experiment still time out; the hot pass remains unidentified.

No confirmed issue draft remains unpublished. This does not assign every remaining
P-256/RSA diagnostic or prove that the named fixes will close all 10 workloads.
Detailed evidence stays on issues/PRs and in ignored raw artifacts, not additional reports.
