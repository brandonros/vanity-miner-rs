# CuMetal status — 2026-09-16

**Historical full 16-workload sweep below: CuMetal `9e3e615`.** The earlier
[three-mode run on `0242f22`](cumetal-three-mode-validation.md) passes Solana,
Bitcoin and Ethereum production with **both LLVM7 and LLVM21 PTX**: 24 invocations,
48 GPU batches, 1,536 CPU-verified candidate positions. #23–25 were closed as
completed on 2026-09-16 after publishing this evidence. No fresh
full-sweep total is claimed. See the [issue matrix](cumetal-issue-matrix.md).

**Current-pin Shallenge production follow-up on `4a207e2`: both LLVM7 and LLVM21
pass all 33 profiles per version** (66 invocations / 132 GPU batches / 4,224
CPU-reference-checked positions / 96 independently verified winner hashes).
[PR #132](https://github.com/Lulzx/cuda-metal/pull/132) implements upstream #129;
its scoped acceptance passes and upstream integration remains pending.
Downstream #38 is closed. See the [Shallenge report](cumetal-shallenge-validation.md).
The other workloads were not rerun at this pin; Shallenge self-test #37 is separate.

RSA source update: production now uses one resumable `kernel_rsa_modulus_vanity`
entry. The RSA GPU measurements below concern the historical four-stage PTX,
not this refactor; the updated RSA self-test also needs fresh GPU validation.
The newer `afe80210` LLVM21 RSA PTX first fails at unsupported `clz.b64`
on measured `c4e5fac`; RSA was not rerun on `4a207e2`. See
[the mining design and checks](gpu-search-pipeline.md).

## Historical results (`9e3e615`)

Measured **2026-09-16, 01:46–02:14 UTC**, using CuMetal **`9e3e61574b77`**
and fresh LLVM 21 PTX from [Actions run #35044328837](https://github.com/brandonros/vanity-miner-rs/actions/runs/35044328837),
producer/host commit **`4e0231a`**. Compiler and runtime hashes were verified.

**2 true, 14 false.** True means the bounded GPU execution and verification passed.
False means translation failed, Metal compilation failed, or the command did not
finish within its **300-second limit**. A timeout does not prove it could never finish.

All failing rows had trackers: **14 workload issues (#23–35 plus #37) were open
at this measurement**. The table records that run, not today's pin or issue states.

| Module | Works in CuMetal | Observed result | Downstream issue |
| --- | --- | --- | --- |
| Solana — production | **False** | PTX conversion/type rejection | [#23](https://github.com/brandonros/vanity-miner-rs/issues/23) |
| Bitcoin — production | **False** | PTX pointer-subtraction rejection | [#24](https://github.com/brandonros/vanity-miner-rs/issues/24) |
| Ethereum — production | **False** | PTX pointer-subtraction rejection | [#25](https://github.com/brandonros/vanity-miner-rs/issues/25) |
| Shallenge — production | **True** | 64 candidates, two verified outputs | — |
| P-256 public key — production | **True** | 64 candidates, two verified outputs; 192 seconds | — |
| P-256 signature — production | **False** | Both search sources fail PTX definedness | [#26](https://github.com/brandonros/vanity-miner-rs/issues/26) |
| RSA modulus — production | **False** | Timeout; late sample in Metal pipeline creation | [#27](https://github.com/brandonros/vanity-miner-rs/issues/27) |
| RSA-PSS — production | **False** | Both search sources time out; sampled in Metal library compilation | [#28](https://github.com/brandonros/vanity-miner-rs/issues/28) |
| Solana — self-test | **False** | PTX definedness in `candidate_match` | [#29](https://github.com/brandonros/vanity-miner-rs/issues/29) |
| Bitcoin — self-test | **False** | PTX definedness in `private_key` | [#30](https://github.com/brandonros/vanity-miner-rs/issues/30) |
| Ethereum — self-test | **False** | PTX pointer subtraction in `candidate_match` | [#31](https://github.com/brandonros/vanity-miner-rs/issues/31) |
| Shallenge — self-test | **False** | PTX definedness in new `sha256_streaming_chunks` check | [#37](https://github.com/brandonros/vanity-miner-rs/issues/37) |
| P-256 public key — self-test | **False** | Generated Metal: missing global and address-space errors | [#32](https://github.com/brandonros/vanity-miner-rs/issues/32) |
| P-256 signature — self-test | **False** | PTX predicate definedness in `low_s` | [#33](https://github.com/brandonros/vanity-miner-rs/issues/33) |
| RSA modulus — self-test | **False** | PTX definedness in new `range_multiple` check | [#34](https://github.com/brandonros/vanity-miner-rs/issues/34) |
| RSA-PSS — self-test | **False** | Generated Metal: invalid pointer cast and malformed global expressions | [#35](https://github.com/brandonros/vanity-miner-rs/issues/35) |

**CPU self-tests: 203 passed. GPU self-tests: 203 blocked before execution.**
Six self-test modules fail PTX translation; two emit Metal that Apple rejects.
No GPU numerical assertion ran. The documented RSA-PSS end-to-end skip was not reached.

The historical report had **3 true / 13 false** and 160 checks. The rewritten
suite has 203 checks; Shallenge grew from 8 to 21 and now fails on a newly added
streaming check. This changed-input result does not establish a pin-only regression.

See the [validation report](cumetal-validation.md) for commands, hashes and exact
diagnostics, and the [issue matrix](cumetal-issue-matrix.md) for implementation
scope and remaining ownership gaps.
