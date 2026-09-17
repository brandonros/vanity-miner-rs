# CuMetal issue matrix

Updated 2026-09-17. **Pin: `075e96330d46331efe2209018b0e47ce9fb485c9`
([PR #154](https://github.com/Lulzx/cuda-metal/pull/154)).** The normal all-mode
miner was rebuilt against the matched Nix compiler/runtime package. The cumulative
implementation PRs remain unmerged. Unpublished #123 and diagnostic PR #128 remain outside this pin.

## Counts and measured progress

- **5 workload issues closed; 11 open:** three production modes and all eight self-test groups.
- **32 combinations:** eight production + eight self-test workloads, each under LLVM7 and LLVM21.
  Ten have accepted results at the revisions below; **22 still lack complete acceptance**.
- Broad #16/#19 make **13 open vanity-miner issues total**. No additional closure this session.
- Bitcoin LLVM7 translation to the same diagnostic fell **97.076 → 22.879 → 8.120 seconds**
  after [printf](https://github.com/Lulzx/cuda-metal/pull/144) and
  [declaration-scan](https://github.com/Lulzx/cuda-metal/pull/148) fixes. No translation cache added.
  [PR #146](https://github.com/Lulzx/cuda-metal/pull/146) adds importer phase timings.
- Shared Release checkpoint at `075e963`: **125 passed, two skipped, zero failed**.
  Two tool-prerequisite tests were excluded; no Debug result is claimed.
- Current-pin Shallenge smoke checks pass both versions: **four GPU batches / 128 CPU-checked
  candidate positions**, two public seeds per version. First launches missed the existing
  pipeline cache; second batches reused their own prepared pipeline. This is not a full profile sweep.

## Fix owners

“Implemented” below describes the stated compiler correction, not full workload acceptance.

| CuMetal owner | Implementation / remaining work |
| --- | --- |
| [#76](https://github.com/Lulzx/cuda-metal/issues/76) | Prior corrections in #131/#135/#142. Original #118 artifact still hits an address-demand proof budget; fresh direct-store range failures need reduction. |
| [#130](https://github.com/Lulzx/cuda-metal/issues/130) | Implemented in #138; P-256 public-key production accepted, downstream #39 closed. |
| [#134](https://github.com/Lulzx/cuda-metal/issues/134) | Implemented in #139; original packed-value blocker cleared. Later pointer proofs remain. |
| [#118](https://github.com/Lulzx/cuda-metal/issues/118) | Per-lane implementation pushed in [#145](https://github.com/Lulzx/cuda-metal/pull/145); 14 GPU fixtures pass. Original full RSA-PSS artifact is blocked earlier by a proof budget. No full-workload resolution claimed. |
| [#140](https://github.com/Lulzx/cuda-metal/issues/140) | Helper-field correction in [#149](https://github.com/Lulzx/cuda-metal/pull/149), compatibility follow-up in #153. Numerical fixtures pass; full LLVM21 Solana still needs a direct-write alias proof. |
| [#136](https://github.com/Lulzx/cuda-metal/issues/136) | Implemented in [#153](https://github.com/Lulzx/cuda-metal/pull/153); disjoint helper writes pass GPU controls. Original Solana/Bitcoin call barriers clear; later direct-store ranges remain unresolved. |
| [#141](https://github.com/Lulzx/cuda-metal/issues/141) | Direct CLZ implemented in [#150](https://github.com/Lulzx/cuda-metal/pull/150); 693 GPU output words pass. RSA advances to POPC or pointer-join failures. |
| [#127](https://github.com/Lulzx/cuda-metal/issues/127) | Immediate PRMT implemented in [#154](https://github.com/Lulzx/cuda-metal/pull/154): at most 21 operations versus 66. Immediate and exhaustive runtime-selector GPU checks pass. Apple time/memory benefit unmeasured. |
| [#133](https://github.com/Lulzx/cuda-metal/issues/133) | Pipeline cost unresolved. LLVM21 Ethereum emits 11.4 MB of MSL at `075e963`; preparation was not rerun. Measure #127's effect separately. |
| [#151](https://github.com/Lulzx/cuda-metal/issues/151) | New dedicated POPC owner/reproducer published; no implementation PR. Next LLVM21 RSA-modulus production blocker. |
| [#152](https://github.com/Lulzx/cuda-metal/issues/152) | New empty-slice length/guard owner and six small controls published; no implementation PR. Confirmed Shallenge case; Bitcoin is a related lead. |

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
| RSA modulus production [#27](https://github.com/brandonros/vanity-miner-rs/issues/27) | Undefined value `c4e5fac` | POPC rejection, dev `463c541` | #151; LLVM7 unresolved |
| RSA-PSS production [#28](https://github.com/brandonros/vanity-miner-rs/issues/28) | Vector parameter rejection `c4e5fac` | CLZ rejection `c4e5fac`, not rerun after fix | Targeted replay needed |
| Solana self-tests [#29](https://github.com/brandonros/vanity-miner-rs/issues/29) | Direct-store range rejection `075e963`, 6.215 s | Direct-write alias rejection `075e963`, 3.732 s | Reduce ranges / #140 alias proof |
| Bitcoin self-tests [#30](https://github.com/brandonros/vanity-miner-rs/issues/30) | Direct-store range rejection `075e963`, 10.850 s | Pointer-type rejection `f7ceeef`, not rerun | Reduce ranges; #118/#152 leads for LLVM21 |
| Ethereum self-tests [#31](https://github.com/brandonros/vanity-miner-rs/issues/31) | Pipeline timeout `0f98856` | Metal emitted, dev `075e963`, 4.812 s; preparation not rerun | #133 |
| Shallenge self-tests [#37](https://github.com/brandonros/vanity-miner-rs/issues/37) | Empty-slice pointer rejection, dev `9c0f340` | Empty-slice pointer rejection, dev `9c0f340` | #152 |
| P-256 public-key self-tests [#32](https://github.com/brandonros/vanity-miner-rs/issues/32) | Undefined value `0f98856` | Metal emitted; GPU unverified `c4e5fac` | Reduce / validate |
| P-256 signature self-tests [#33](https://github.com/brandonros/vanity-miner-rs/issues/33) | Undefined value `0f98856` | Undefined value `c4e5fac` | Reduce |
| RSA modulus self-tests [#34](https://github.com/brandonros/vanity-miner-rs/issues/34) | Undefined value `c4e5fac` | CLZ rejection `c4e5fac`, not rerun after fix | #151 occurs in input; first failure needs replay |
| RSA-PSS self-tests [#35](https://github.com/brandonros/vanity-miner-rs/issues/35) | Pointer/integer join rejection, dev `463c541` | CLZ rejection `c4e5fac`, not rerun after fix | Reduce join; original mixed-lane artifact remains on #118 |

Solana/Bitcoin failures above occur before GPU launch: 79/39 checks are blocked,
not numerical assertion failures. Current inputs remain producer `afe80210`,
[Actions 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
Consumer `67950b7` contains the same logic and kernel entries after the crate move.

## Next work

1. Implement the small #151 opcode gap; replay only affected RSA inputs.
2. Implement #152's bounded zero-length memory/guard proof; keep integer sentinels intact.
3. Reduce the new direct-store range/alias failures before assigning duplicate owners or widening budgets.
4. Measure #127 against #133 with separate library/pipeline timings. Smaller source alone is insufficient.

No confirmed issue draft remains unpublished. This does not assign every remaining
P-256/RSA diagnostic or prove that the named fixes will close all 11 workloads.
Detailed evidence stays on issues/PRs and in ignored raw artifacts, not additional reports.
