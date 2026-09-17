# CuMetal issue matrix

Updated 2026-09-17. **Pin: `0dbdb4bc89bc8d7e1d622686c2f08b9ea9cb5afc`
([PR #160](https://github.com/Lulzx/cuda-metal/pull/160)).** The normal all-mode
miner was rebuilt against the matched Nix compiler/runtime package. The cumulative
implementation PRs remain unmerged. #155/#156/#158/#159/#160 are included. Unpublished #123 and diagnostic PR #128 remain outside this pin.

## Counts and measured progress

- **5 workload issues closed; 11 open:** three production modes and all eight self-test groups.
- **32 combinations:** eight production + eight self-test workloads, each under LLVM7 and LLVM21.
  Ten have accepted results at the revisions below; **22 still lack complete acceptance**.
- Broad #16/#19 make **13 open vanity-miner issues total**. No additional closure this session.
- Bitcoin LLVM7 translation to the same diagnostic fell **97.076 → 22.879 → 8.120 seconds**
  after [printf](https://github.com/Lulzx/cuda-metal/pull/144) and
  [declaration-scan](https://github.com/Lulzx/cuda-metal/pull/148) fixes. No translation cache added.
  [PR #146](https://github.com/Lulzx/cuda-metal/pull/146) adds importer phase timings.
- Combined Release checkpoint: **128 passed, two skipped**, with two tool-prerequisite
  tests excluded, at development `04a0955`. The final generic-store scope narrowing
  at `0dbdb4b` passes all five affected suites. No Debug result is claimed.
- Fresh normal-pin Shallenge self-tests: CPU **21/21 pass**; both GPU versions translate
  but Apple rejects two unqualified pointer casts per module. All 21 checks are blocked
  before execution, not numerical mismatches. [Published evidence](https://github.com/brandonros/vanity-miner-rs/issues/37#issuecomment-5718953337).
- Shallenge production's earlier `075e963` smoke checks passed both versions:
  four GPU batches / 128 CPU-checked candidate positions. They were not rerun at this pin.

## Fix owners

“Implemented” below describes the stated compiler correction, not full workload acceptance.

| CuMetal owner | Implementation / remaining work |
| --- | --- |
| [#76](https://github.com/Lulzx/cuda-metal/issues/76) | Prior #131/#135/#142 corrections plus demand scaling in [#158](https://github.com/Lulzx/cuda-metal/pull/158). Original #118 input clears the budget stage at `0dbdb4b`, then rejects an unresolved store offset (line134556, cell64); reduction remains. |
| [#130](https://github.com/Lulzx/cuda-metal/issues/130) | Implemented in #138; P-256 public-key production accepted, downstream #39 closed. |
| [#134](https://github.com/Lulzx/cuda-metal/issues/134) | Implemented in #139; original packed-value blocker cleared. Later pointer proofs remain. |
| [#118](https://github.com/Lulzx/cuda-metal/issues/118) | Per-lane correction in [#145](https://github.com/Lulzx/cuda-metal/pull/145); 17 mixed-lane GPU fixtures pass in the latest group. Original full artifact now reaches a later store-range proof; current LLVM21 RSA-PSS self-tests emit MSL. Neither is a full GPU acceptance claim. |
| [#140](https://github.com/Lulzx/cuda-metal/issues/140) | Helper-field correction in [#149](https://github.com/Lulzx/cuda-metal/pull/149), compatibility in #153/#155/#158. Latest measured LLVM21 Solana rejects an empty helper record whose reaching store lacks a pointer type. Context-sensitive length/pointer correlation remains. |
| [#136](https://github.com/Lulzx/cuda-metal/issues/136) | Implemented in [#153](https://github.com/Lulzx/cuda-metal/pull/153); disjoint helper writes pass GPU controls. Original Solana/Bitcoin call barriers clear; later direct-store ranges remain unresolved. |
| [#141](https://github.com/Lulzx/cuda-metal/issues/141) | CLZ in [#150](https://github.com/Lulzx/cuda-metal/pull/150); trap-capable helper support in [#159](https://github.com/Lulzx/cuda-metal/pull/159). Current LLVM21 RSA-PSS production/self-test modules emit MSL. |
| [#127](https://github.com/Lulzx/cuda-metal/issues/127) | Immediate PRMT implemented in [#154](https://github.com/Lulzx/cuda-metal/pull/154): at most 21 operations versus 66. Immediate and exhaustive runtime-selector GPU checks pass. Apple time/memory benefit unmeasured. |
| [#133](https://github.com/Lulzx/cuda-metal/issues/133) | Pipeline cost unresolved. LLVM21 Ethereum emits 11.4 MB of MSL at `075e963`; preparation was not rerun. Measure #127's effect separately. |
| [#151](https://github.com/Lulzx/cuda-metal/issues/151) | POPC implemented in [#156](https://github.com/Lulzx/cuda-metal/pull/156). Shared CLZ/POPC numerical tests verify 6,150 output words. RSA modulus advances to undefined incoming-register failures. |
| [#152](https://github.com/Lulzx/cuda-metal/issues/152) | Bounded local zero-length proof in [#158](https://github.com/Lulzx/cuda-metal/pull/158). Both Shallenge inputs translate with #160, then hit Metal qualifier errors in streaming-SHA helpers. Further guard-propagation ownership is under investigation. |
| [#157](https://github.com/Lulzx/cuda-metal/issues/157) | Generic-store stale cell types fixed in [#160](https://github.com/Lulzx/cuda-metal/pull/160); scalar/vector numerical and refusal checks pass. This clears LLVM7 Shallenge’s pointer/scalar join. General #137 escape activation is separate. |

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
| RSA modulus production [#27](https://github.com/brandonros/vanity-miner-rs/issues/27) | Undefined value `c4e5fac` | Undefined `%rd205`, dev `400a8bb`, 9.175 s | Reduce undefined edge; LLVM7 unresolved |
| RSA-PSS production [#28](https://github.com/brandonros/vanity-miner-rs/issues/28) | Vector parameter rejection `c4e5fac` | MSL emitted, dev `0dbdb4b`, 65.502 s | Apple preparation / GPU validation; historical #115 |
| Solana self-tests [#29](https://github.com/brandonros/vanity-miner-rs/issues/29) | Direct-store range rejection `075e963`, 6.215 s | Helper-field type refusal at line50529, dev `70e2ab0`, 3.936 s | Reduce LLVM7 ranges / #140 empty-record context |
| Bitcoin self-tests [#30](https://github.com/brandonros/vanity-miner-rs/issues/30) | Direct-store range rejection `075e963`, 10.850 s | Pointer-type rejection `f7ceeef`, not rerun | Reduce ranges; #118/#152 leads for LLVM21 |
| Ethereum self-tests [#31](https://github.com/brandonros/vanity-miner-rs/issues/31) | Pipeline timeout `0f98856` | Metal emitted, dev `075e963`, 4.812 s; preparation not rerun | #133 |
| Shallenge self-tests [#37](https://github.com/brandonros/vanity-miner-rs/issues/37) | Apple qualifier errors `0dbdb4b`, 12.791 s | Apple qualifier errors `0dbdb4b`, 3.841 s | Streaming boundary/chunks; guard-propagation reduction needed |
| P-256 public-key self-tests [#32](https://github.com/brandonros/vanity-miner-rs/issues/32) | Undefined value `0f98856` | Metal emitted; GPU unverified `c4e5fac` | Reduce / validate |
| P-256 signature self-tests [#33](https://github.com/brandonros/vanity-miner-rs/issues/33) | Undefined value `0f98856` | Undefined value `c4e5fac` | Reduce |
| RSA modulus self-tests [#34](https://github.com/brandonros/vanity-miner-rs/issues/34) | Undefined value `c4e5fac` | Undefined `%rd2065`, dev `400a8bb`, 17.594 s | Reduce undefined edge |
| RSA-PSS self-tests [#35](https://github.com/brandonros/vanity-miner-rs/issues/35) | Pointer/integer join rejection, dev `463c541` | MSL emitted, dev `0dbdb4b`, 93.728 s | Apple preparation / GPU checks; original #118 artifact still range-blocked |

Solana/Bitcoin failures above occur before GPU launch: 79/39 checks are blocked,
not numerical assertion failures. Current inputs remain producer `afe80210`,
[Actions 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
Consumer `67950b7` contains the same logic and kernel entries after the crate move.

## Next work

1. Reduce Shallenge’s two streaming-SHA qualifier failures; distinguish guard propagation
   from the separately tracked #137 scalar-load activation gap.
2. Reduce the remaining direct-store ranges and #140 empty-record context; retain the
   original #118 artifact as a separate acceptance input.
3. Advance the now-translating LLVM21 RSA-PSS inputs through Apple preparation and numerical checks.
4. Measure #127 against #133 with separate library/pipeline timings. Smaller source alone is insufficient.

No confirmed issue draft remains unpublished. This does not assign every remaining
P-256/RSA diagnostic or prove that the named fixes will close all 11 workloads.
Detailed evidence stays on issues/PRs and in ignored raw artifacts, not additional reports.
