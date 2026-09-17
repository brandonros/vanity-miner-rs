# CuMetal issue matrix

Updated 2026-09-17 from existing measurements and GitHub; no new build or sweep.
**Pin: `fb3644a251befdc4c1a9d82a8742ae04ae2498c4` ([PR #142](https://github.com/Lulzx/cuda-metal/pull/142)).**
This contains the published cumulative fix stack. #118 and #123 have unpublished
work outside it; diagnostic PR #128 is also on a separate branch.

## Counts

- **5 workload issues closed, 11 open:** three production modes and all eight self-test groups.
- **32 combinations:** 8 production + 8 self-test workloads, each under LLVM7 and LLVM21.
- **10 combinations accepted at the revisions below; 22 lack complete acceptance.**
- Broad #16/#19 make **13 open vanity-miner issues total**. No additional closure is justified.

## Published work and remaining implementation

The latest overnight stretch published **one additional fix**, #76 scaling in
PR #142. Including the preceding work, **three issue scopes have published
fixes: #76, #130 and #134**. Their PRs are unmerged; a cleared compiler stage
alone does not establish a working workload.

| CuMetal issue | State | PR / remaining work |
| --- | --- | --- |
| [#76 — types and memory proofs](https://github.com/Lulzx/cuda-metal/issues/76) | Known corrections implemented and pushed | [#131](https://github.com/Lulzx/cuda-metal/pull/131), [#135](https://github.com/Lulzx/cuda-metal/pull/135), [#142](https://github.com/Lulzx/cuda-metal/pull/142). Native legacy validation and integration remain. |
| [#130 — scalar zero guards](https://github.com/Lulzx/cuda-metal/issues/130) | Implemented; P-256 public-key production verified | [#138](https://github.com/Lulzx/cuda-metal/pull/138); downstream #39 closed. |
| [#134 — packed-value narrowing](https://github.com/Lulzx/cuda-metal/issues/134) | Implemented; original Bitcoin blocker cleared | [#139](https://github.com/Lulzx/cuda-metal/pull/139); later pointer failures remain. |
| [#118 — mixed vector lanes](https://github.com/Lulzx/cuda-metal/issues/118) | Local implementation in progress; no PR | Initial unit/translation controls pass. GPU and full-input validation unfinished. |
| [#140 — helper pointer provenance](https://github.com/Lulzx/cuda-metal/issues/140) | Researched; not implemented | Confirmed Solana numerical failure. |
| [#136 — helper write footprints](https://github.com/Lulzx/cuda-metal/issues/136) | Researched; not implemented | Solana/Bitcoin pointer-cell proofs. |
| [#141 — direct CLZ](https://github.com/Lulzx/cuda-metal/issues/141) | Issue/reproducer published; not implemented | RSA `clz.b32` / `clz.b64` support. |
| [#133 — Metal pipeline cost](https://github.com/Lulzx/cuda-metal/issues/133) | Measured/researched; no demonstrated fix | Ethereum self-test preparation timeout. |

Implementation is paused for cleanup. Resume order: **#118 → #140 → #136 → #141 → #133**.
**Zero confirmed new issue drafts need publication now.** These five owners already
exist. Other P-256/RSA diagnostics and Bitcoin's empty-slice case still need
research; this is not a claim that five fixes will resolve all 22 combinations.

## Workloads: latest recorded result per LLVM version

**Pass means GPU execution plus CPU checks.** Translation success alone is not a
pass. Each cell names its measured CuMetal revision; historical passes are not
being relabeled as current-pin reruns.

| Workload / vanity-miner issue | LLVM7 | LLVM21 | Next CuMetal owner |
| --- | --- | --- | --- |
| Solana production [#23](https://github.com/brandonros/vanity-miner-rs/issues/23), closed | Pass `0242f22` | Pass `0242f22` | — |
| Bitcoin production [#24](https://github.com/brandonros/vanity-miner-rs/issues/24), closed | Pass `0242f22` | Pass `0242f22` | — |
| Ethereum production [#25](https://github.com/brandonros/vanity-miner-rs/issues/25), closed | Pass `0242f22` | Pass `0242f22` | — |
| Shallenge production [#38](https://github.com/brandonros/vanity-miner-rs/issues/38), closed | Pass `4a207e2` | Pass `4a207e2` | — |
| P-256 public-key production [#39](https://github.com/brandonros/vanity-miner-rs/issues/39), closed | Pass `0f98856` | Pass `0f98856` | — |
| P-256 signature production [#26](https://github.com/brandonros/vanity-miner-rs/issues/26) | Vector parameter rejection `13efc29` | Undefined value `c4e5fac` | Needs reduction; #41 is a lead |
| RSA modulus production [#27](https://github.com/brandonros/vanity-miner-rs/issues/27) | Undefined value `c4e5fac` | CLZ rejection `c4e5fac` | #141 for LLVM21; LLVM7 unresolved |
| RSA-PSS production [#28](https://github.com/brandonros/vanity-miner-rs/issues/28) | Vector parameter rejection `c4e5fac` | CLZ rejection `c4e5fac` | #141 for LLVM21; LLVM7 unresolved |
| Solana self-tests [#29](https://github.com/brandonros/vanity-miner-rs/issues/29) | Helper-write rejection `13efc29` | 78 pass / 1 numerical failure `0f98856` | #136 / #140 |
| Bitcoin self-tests [#30](https://github.com/brandonros/vanity-miner-rs/issues/30) | Helper-write rejection `fb3644a` | Pointer-type rejection `f7ceeef` | #136 / #118; empty-slice proof also pending |
| Ethereum self-tests [#31](https://github.com/brandonros/vanity-miner-rs/issues/31) | Pipeline timeout `0f98856` | Pipeline timeout `4a207e2` | #133 |
| Shallenge self-tests [#37](https://github.com/brandonros/vanity-miner-rs/issues/37) | Mixed-vector rejection `13efc29` | Metal pointer errors `0f98856` | #118 |
| P-256 public-key self-tests [#32](https://github.com/brandonros/vanity-miner-rs/issues/32) | Undefined value `0f98856` | Metal emitted; GPU unverified `c4e5fac` | Needs fresh reduction/validation |
| P-256 signature self-tests [#33](https://github.com/brandonros/vanity-miner-rs/issues/33) | Undefined value `0f98856` | Undefined value `c4e5fac` | Needs reduction |
| RSA modulus self-tests [#34](https://github.com/brandonros/vanity-miner-rs/issues/34) | Undefined value `c4e5fac` | CLZ rejection `c4e5fac` | #141 for LLVM21; LLVM7 unresolved |
| RSA-PSS self-tests [#35](https://github.com/brandonros/vanity-miner-rs/issues/35) | CLZ rejection `fb3644a` | CLZ rejection `c4e5fac` | #141; original mixed-lane reproducer stays on #118 |

Latest normal-consumer evidence at `fb3644a`: **53 CPU checks pass**; Bitcoin
LLVM7 stops at #136 after **94.244 s**, RSA-PSS LLVM7 at #141 after **80.859 s**.
No full-module GPU launch occurs. [Published evidence and exact identities](https://github.com/Lulzx/cuda-metal/issues/76#issuecomment-5713697751).
Current corpus: producer `afe80210`, [Actions 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
Keep detailed history/reproducers on the linked issues and PRs, not in this file.
