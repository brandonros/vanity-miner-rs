# CuMetal issue ownership and fix status

RSA source update: production now uses one resumable `kernel_rsa_modulus_vanity`
entry. The RSA GPU measurements below concern the historical four-stage PTX,
not this refactor; the updated RSA self-test also needs fresh GPU validation.
See [the mining design](gpu-search-pipeline.md).

## New split-bundle translation check

The [16-row, two-version comparison](cumetal-ptx-bundle-review.md) checks the new
`7484a5d` PTX bundles against CuMetal `9e3e615`: **LLVM 7 emits Metal for 2/16
modules; LLVM 21 for 6/16**. These are 32 translation attempts, without Apple
compilation or GPU execution. RSA modulus selects only its first entry.

LLVM 21 Solana/Bitcoin/Ethereum reproduce the #76 first diagnostics. LLVM 7
Bitcoin/Ethereum emit Metal; LLVM 7 Solana has a conversion/join-shaped `mul.hi`
failure that needs confirmation before sharing #76 ownership. LLVM 7 also adds
Shallenge and P-256 public-key production failures, now tracked in
[#38](https://github.com/brandonros/vanity-miner-rs/issues/38) and
[#39](https://github.com/brandonros/vanity-miner-rs/issues/39). See the comparison for all results and scope limits.

The [#76 implementation plan](cumetal-76-plan.md) defines the code changes and
completion gates. The full GPU results below remain tied to their earlier
producer and input hashes; translation counts do not replace them.

### Measured translation times by LLVM version

Recorded from `.cumetal-artifacts/ptx-compile-ftsphb4j/timings.tsv`, run started
**2026-09-16 02:45:36 UTC**. Each cell gives the result and elapsed seconds for
one serial compiler invocation. **Emit** means Metal source was written;
**fail** means the compiler rejected the input. No attempt timed out under the
300-second per-file limit. Fail times measure time to rejection, so these totals
are not a comparison of successful compilation performance.

| Module | LLVM 7 (seconds) | LLVM 21 (seconds) | Published tracker/evidence |
| --- | --- | --- | --- |
| `solana` | Fail 4.587 | Fail 4.792 | [#23 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/23#issuecomment-5691525016) |
| `bitcoin` | Emit 17.027 | Fail 11.171 | [#24 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/24#issuecomment-5691525247) |
| `ethereum` | Emit 15.252 | Fail 10.365 | [#25 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/25#issuecomment-5691525511) |
| `shallenge` | Fail 0.350 | Emit 0.409 | [#38 — new issue](https://github.com/brandonros/vanity-miner-rs/issues/38) |
| `p256_public_key` | Fail 9.412 | Emit 6.906 | [#39 — new issue](https://github.com/brandonros/vanity-miner-rs/issues/39) |
| `p256_signature` | Fail 53.571 | Fail 20.222 | [#26 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/26#issuecomment-5691525737) |
| `rsa_modulus` | Fail 13.861 | Emit 9.809 | [#27 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/27#issuecomment-5691526032); advance entry only |
| `rsa_pss` | Fail 22.353 | Emit 71.486 | [#28 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/28#issuecomment-5691526316) |
| `self_test_solana` | Fail 152.788 | Fail 69.859 | [#29 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/29#issuecomment-5691526564) |
| `self_test_bitcoin` | Fail 85.895 | Fail 59.725 | [#30 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/30#issuecomment-5691526792) |
| `self_test_ethereum` | Fail 18.879 | Fail 20.568 | [#31 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/31#issuecomment-5691527077) |
| `self_test_shallenge` | Fail 26.332 | Fail 13.648 | [#37 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/37#issuecomment-5691528582) |
| `self_test_p256_public_key` | Fail 25.399 | Emit 23.656 | [#32 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/32#issuecomment-5691527538) |
| `self_test_p256_signature` | Fail 124.183 | Fail 48.627 | [#33 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/33#issuecomment-5691527877) |
| `self_test_rsa_modulus` | Fail 124.444 | Fail 93.508 | [#34 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/34#issuecomment-5691528097) |
| `self_test_rsa_pss` | Fail 57.677 | Emit 159.353 | [#35 — comparison comment](https://github.com/brandonros/vanity-miner-rs/issues/35#issuecomment-5691528345) |
| **Total** | **2 emit / 14 fail; 752.010 s (12m 32s)** | **6 emit / 10 fail; 624.104 s (10m 24s)** | **32 attempts; 24 translation failures** |

The [bundle comparison](cumetal-ptx-bundle-review.md) records the first diagnostics,
compiler/runtime hashes, bundle hashes and reproduction command. The local
`report.md` contains all failure text and log links; `timings.tsv` records each
input's SHA-256. Use [the timed runner](../scripts/compile-ptx-cumetal.py) to repeat
the sweep. Its default entry selection covers only `kernel_rsa_advance` in the
four-entry RSA modulus file: 32 file attempts are **not** 38-entry coverage.

### GitHub tracking decision

Published on **2026-09-16**: [#38 — LLVM 7 Shallenge production](https://github.com/brandonros/vanity-miner-rs/issues/38)
and [#39 — LLVM 7 P-256 public-key production](https://github.com/brandonros/vanity-miner-rs/issues/39).
Both issues were read back and verified open. Together with **#23–35 and #37**,
there are now **16 open workload trackers**, one for each logical module.
LLVM 7/21 comparison comments were published and read back on all **14 existing
trackers (#23–35 and #37)**. The timing table links directly to each verified
comment; previous issue bodies and full-GPU histories are preserved. No workload
was closed by this translation-only evidence.

**Keep one downstream issue per workload, with separate LLVM 7 and LLVM 21
evidence and acceptance rows.** The 24 failed invocations are not 24 established
compiler defects. The 14 comparison comments add this run's version-specific
first failures or translation successes without replacing the earlier full-GPU
evidence. Record translation, Apple compilation, pipeline
preparation, GPU completion and CPU verification separately. MSL emission in
one version does not close the other version's failure or the workload issue.

**New downstream workload issues:**

| Published issue | Exact new evidence | Comparison and scope |
| --- | --- | --- |
| [#38 — LLVM 7 Shallenge production](https://github.com/brandonros/vanity-miner-rs/issues/38) | `shallenge.ptx`: exit 1 after 0.350 s; line 306, `pointer subtraction requires a pointer minus a 64-bit integer byte offset`. SHA-256 `bfbd79fec26057bfc77293777f3acbcb40cb84c61041274e2c66fe6770639e51`. | LLVM 21 emits MSL in 0.409 s on this bundle. Production is distinct from self-test issue #37. Upstream owner unconfirmed; similar pointer-offset wording alone does not establish #76 ownership. |
| [#39 — LLVM 7 P-256 public-key production](https://github.com/brandonros/vanity-miner-rs/issues/39) | `p256_public_key.ptx`: exit 1 after 9.412 s; `%rd18761` undefined on an incoming edge to `$L__BB5_1`. SHA-256 `dbd5ada9e96613eb7f2da98e97b95310a5a6368744c5f8d1d84b9c184184e0da`. | LLVM 21 emits MSL in 6.906 s on this bundle. Production is distinct from self-test issue #32. Upstream owner unconfirmed; reduce this join before assigning a CFG/predicate issue. |

Both published issues retain producer `7484a5d` plus the recorded timing
instrumentation, CuMetal `9e3e615`, the exact compiler and input hashes, and the
full diagnostic. First require the retained LLVM 7 input to translate, then
complete Apple compilation and bounded CPU-verified production batches with
input/buffer guards intact; retain an independent LLVM 21 regression row. A
compiler correction or reduced reproducer belongs upstream once its scope is
confirmed, linked back to this downstream workload tracker. No upstream issue
assignment is established by these new rows.

The issue bodies include the producer revision, build/translation commands,
exact diagnostics and input hashes. Original `.cumetal-artifacts/` snapshots
remain local evidence, not public attachments; regenerated inputs must have
their hashes checked before being attributed to this run. Do not substitute
the older Actions-produced inputs for these locally generated bundles.

## Recorded full GPU run

Updated **2026-09-16** after the complete fresh run, **01:46:07–02:14:38 UTC**.
The [16-row result](cumetal-status.md) is **2 true / 14 false**: Shallenge and
P-256 public-key production pass; all eight self-test groups are blocked before
execution. That report had **14 workload issues (#23–35 plus #37)**, including
[#37 — Shallenge self-tests](https://github.com/brandonros/vanity-miner-rs/issues/37).
The subsequent LLVM 7 production trackers **#38/#39** bring the current total to
**16**; they do not change the earlier full-GPU measurements.
This tracking update adds no new measurements.

## Tested source and pin

- **CuMetal:** `9e3e61574b776424a96c686bdbdc04ad1f27fe9f` from `flake.lock`,
  the head of [PR #122](https://github.com/Lulzx/cuda-metal/pull/122).
  Git ancestry includes #114, #117, #121 and their prerequisite stack.
- **Consumer and PTX producer:** `4e0231aa82b69be146936f53829c95ef3f522832`.
  Fresh LLVM 21 PTX came from [GitHub Actions run 35044328837](https://github.com/brandonros/vanity-miner-rs/actions/runs/35044328837).
  The CLI was built from that exact source, with the locked compiler/runtime
  hashes checked on every command.
- **Coverage:** 18 commands represent eight production modes and eight self-test
  groups. Production has **2 passing modes, 4 translation-blocked modes, and
  2 timed-out modes**. The self-tests have **0 GPU passes, 203 blocked checks,
  0 numerical failures, 0 skips, and 0 missing outcomes**. The matching CPU build
  passes all **203** checks. See [validation details](cumetal-validation.md).
- **Outside this pin:** [diagnostic tracing PR #128](https://github.com/Lulzx/cuda-metal/pull/128)
  at `6b70a1b618c4ab1d80e094991941aa56970aff39`, plus unpublished #123 work.
  The audit found no newer implementation PR; downstream Shallenge tracking was
  subsequently added as #37.

The historical complete run at `e5acf8cc0c65` had **3 true / 13 false** and 160
self-test checks. The current checks were reorganized and expanded to 203.
In particular, Shallenge grew from eight checks to 21 and now fails translation
inside a newly added streaming-SHA check. This changed-input result does not by
itself establish a CuMetal compiler regression.

All eight production PTX files match the old complete text after a consistent,
one-to-one renaming of module symbols and their parameter names; instructions,
registers, labels and data bytes are unchanged. The self-test PTX files have
substantive changes. Symbol changes can still affect translation or cache reuse;
this comparison does not establish identical compilation time or GPU behavior.

## Read the statuses separately

| Status | What it establishes |
| --- | --- |
| **No published fix PR** | The public audit found no patch for that remaining case. It does not prove nobody has started locally. |
| **Local WIP; no PR** | An explicitly reported unpublished implementation exists outside the pin. |
| **PR included** | The locked contribution stack contains the code. The relevant PRs remain open drafts, rather than merged upstream fixes. |
| **Scoped blocker cleared** | An identified original artifact passes the fixed stage. A later failure, or a different input's error, does not erase that evidence. |
| **Research lead; owner unconfirmed** | Existing issue scope resembles the fresh diagnostic. A reducer and identified implementation test are still needed before declaring a duplicate or expanding acceptance. |
| **Workload verified** | The identified workload completes GPU execution and its required result/guard/CPU checks. Emitted Metal or successful module-handle creation is insufficient. |

## Downstream-to-upstream matrix

**Evidence is from the recorded full GPU run unless marked split-bundle.** Existing upstream owners retain their
original reproducer scope. Newly observed diagnostics are assigned only where
supported; research leads are labeled explicitly. No blocked self-test row is
an individual numerical assertion failure.

| Vanity-miner issue | Upstream ownership | Published implementation | Included in `9e3e615`? | Fresh result and next work |
| --- | --- | --- | --- | --- |
| [#23 — Solana production](https://github.com/brandonros/vanity-miner-rs/issues/23) | [#76 — conversion/join types](https://github.com/Lulzx/cuda-metal/issues/76) | **No PR for the remaining destination/normalized-definition/join work.** #50/#56/#75 cover narrower cases. | Earlier PRs only. | **PTX translation fails:** first `i32`/`i64` disagreement at line 34070. Implement type inference/propagation, then retest verified batches. |
| [#24 — Bitcoin production](https://github.com/brandonros/vanity-miner-rs/issues/24) | [#76](https://github.com/Lulzx/cuda-metal/issues/76) | **No PR for the remaining gap.** Shared work with #23/#25. | Earlier PRs only. | **PTX translation fails:** pointer subtraction at line 26556. Same first diagnostic as the historical artifact; separate downstream verification remains required. |
| [#25 — Ethereum production](https://github.com/brandonros/vanity-miner-rs/issues/25) | [#76](https://github.com/Lulzx/cuda-metal/issues/76) | **No PR for the remaining gap.** Shared work with #23/#24. | Earlier PRs only. | **PTX translation fails:** pointer subtraction at line 22069. Implement the shared correction and verify this workload independently. |
| [#26 — P-256 signature production](https://github.com/brandonros/vanity-miner-rs/issues/26) | [#83 — empty labels/predicate facts](https://github.com/Lulzx/cuda-metal/issues/83) | #85 covers original predicate cases; **no empty-label follow-up PR**. | #85 and #121 included; #121 addresses a different depth case. | **Both sources fail PTX translation:** `%rd17653` undefined at `$L__BB0_11`. Preserve facts across empty blocks and validate message/ephemeral search. |
| [#27 — RSA modulus production](https://github.com/brandonros/vanity-miner-rs/issues/27) | [#119](https://github.com/Lulzx/cuda-metal/issues/119) owns the **historical GPU-completion wait**. Fresh pipeline-preparation wait: **owner unconfirmed**. | **No completion/performance fix PR.** | No corresponding fix. | **Times out:** fresh samples at about 180 s and near the deadline are inside **Metal pipeline creation**, before verified execution. This is not the historical device-synchronization sample. Attribute the preparation cost and confirm an owner; then complete verified four-stage cycles. |
| [#28 — RSA-PSS production](https://github.com/brandonros/vanity-miner-rs/issues/28) | Earlier #96/#109/#111 → [#115 — Metal compilation time](https://github.com/Lulzx/cuda-metal/issues/115) | #108/#110/#112 clear translation; [#128](https://github.com/Lulzx/cuda-metal/pull/128) adds **diagnostics only**. **No performance fix PR.** | Translation fixes yes; #128 **no**. | **Salt and message searches time out** while sampled in Metal source-library compilation. Both emit source; neither verifies candidates. Diagnose and reduce compilation cost, then validate both sources. |
| [#29 — Solana self-tests](https://github.com/brandonros/vanity-miner-rs/issues/29) | Original [#46](https://github.com/Lulzx/cuda-metal/issues/46)/[#116](https://github.com/Lulzx/cuda-metal/issues/116); fresh CFG error **unassigned**, [#83](https://github.com/Lulzx/cuda-metal/issues/83)/[#35](https://github.com/Lulzx/cuda-metal/issues/35) are leads. | [#117](https://github.com/Lulzx/cuda-metal/pull/117) clears the original trap-helper blocker. **No demonstrated fix for this new case.** | #117 **yes**. | **79 checks blocked by translation:** `%rd42` undefined at `$L__BB77_6` inside new `solana.candidate_match`. Reduce the guarded-use path; do not label this the old trap-expansion failure. |
| [#30 — Bitcoin self-tests](https://github.com/brandonros/vanity-miner-rs/issues/30) | Original [#113](https://github.com/Lulzx/cuda-metal/issues/113)/[#116](https://github.com/Lulzx/cuda-metal/issues/116); fresh error **unassigned**, [#51](https://github.com/Lulzx/cuda-metal/issues/51)/[#70](https://github.com/Lulzx/cuda-metal/pull/70) discarded-half scope is a lead. | [#114](https://github.com/Lulzx/cuda-metal/pull/114), [#117](https://github.com/Lulzx/cuda-metal/pull/117). **No demonstrated fix for the fresh case.** | **Yes**, existing scoped fixes. | **39 checks blocked by translation:** `%r1688` undefined at `$L__BB4_3`, in outlined `bitcoin.private_key`. It supplies a packed high half later narrowed to low 16 bits. Confirm unused-bit normalization before assigning ownership; the older Metal wait is not this run's first failure. |
| [#31 — Ethereum self-tests](https://github.com/brandonros/vanity-miner-rs/issues/31) | Original [#46](https://github.com/Lulzx/cuda-metal/issues/46)/[#116](https://github.com/Lulzx/cuda-metal/issues/116); fresh [#76](https://github.com/Lulzx/cuda-metal/issues/76) type-provenance scope is a **lead**. | [#117](https://github.com/Lulzx/cuda-metal/pull/117). **No demonstrated correction for this new case.** | #117 **yes**. | **8 checks blocked by translation:** pointer subtraction at line 17373 inside new `ethereum.candidate_match`. Reduce the suffix-offset types; matching the production diagnostic alone does not prove the same cause. |
| [#32 — P-256 public-key self-tests](https://github.com/brandonros/vanity-miner-rs/issues/32) | Original [#116](https://github.com/Lulzx/cuda-metal/issues/116); fresh helper-global scope [#123](https://github.com/Lulzx/cuda-metal/issues/123) is a **lead**; additional address-space error **unassigned**. | [#117](https://github.com/Lulzx/cuda-metal/pull/117); #123 has unpublished WIP. | #117 **yes**; #123 **no**. | **13 checks blocked by Apple Metal compilation:** undeclared `private$39`, plus a pointer cast missing an explicit address space. PTX references the global in HMAC/scalar/end-to-end helpers. Test both error families; registration issue #125 is not established by these compilation errors. |
| [#33 — P-256 signature self-tests](https://github.com/brandonros/vanity-miner-rs/issues/33) | Original [#120](https://github.com/Lulzx/cuda-metal/issues/120); fresh CFG error **unassigned**, [#83](https://github.com/Lulzx/cuda-metal/issues/83)/#120 are leads. | [#121](https://github.com/Lulzx/cuda-metal/pull/121) clears the original eight-block-depth case. **No demonstrated fresh-case fix.** | #121 **yes**. | **13 checks blocked by translation:** `%p243` undefined at `$L__BB8_1` inside outlined `p256_signature.low_s`. Reduce the new validity-guard path; this is not proof the original `%rs506` correction regressed. |
| [#34 — RSA modulus self-tests](https://github.com/brandonros/vanity-miner-rs/issues/34) | Original [cuda-metal #35](https://github.com/Lulzx/cuda-metal/issues/35) → [#124](https://github.com/Lulzx/cuda-metal/issues/124), with optimization [#127](https://github.com/Lulzx/cuda-metal/issues/127). Fresh masked-payload error: **#35 scope lead**, coverage unconfirmed. | [#122](https://github.com/Lulzx/cuda-metal/pull/122) clears the original masked-payload SSA. **No resource/PRMT fix PR or demonstrated new-case fix.** | #122 **yes**. | **16 checks blocked by translation:** `%rs907` undefined at `$L__BB16_1` inside new `rsa_modulus.range_multiple`. The comparison is masked by a presence predicate; reduce it before extending #35 acceptance. This run never reaches the original artifact's Metal allocation failure. |
| [#35 — RSA-PSS self-tests](https://github.com/brandonros/vanity-miner-rs/issues/35) | Original [#118 — mixed pointer lanes](https://github.com/Lulzx/cuda-metal/issues/118) remains open. Fresh pointer-cast error **unassigned**; later raw-global expressions are a [#123](https://github.com/Lulzx/cuda-metal/issues/123)-adjacent **lead**. | **No PR for #118; no demonstrated fix for the fresh Metal errors.** #108/#114/#117 cover other cases. | Related fixes yes; missing work **no**. | **14 checks blocked by Apple Metal compilation:** invalid `as_type<device uchar*>` from `ulong`, then syntax errors from unlowered `[private$em]` expressions. The latter symbol occurs in CRT helpers. Changed PTX reaching Metal does not demonstrate #118 fixed on its original reproducer. |
| [#37 — Shallenge self-tests](https://github.com/brandonros/vanity-miner-rs/issues/37) | **No confirmed upstream owner.** [#83](https://github.com/Lulzx/cuda-metal/issues/83) predicate/definedness scope is a lead. | **No demonstrated fix PR.** | No identified correction. | **21 checks blocked by translation:** `%rd16` undefined at `$L__BB18_1` inside new `shallenge.sha256_streaming_chunks`. Reduce the empty/nonempty chunk guard and assign ownership. The historical eight-check pass does not cover this expanded module. |
| [#38 — LLVM 7 Shallenge production](https://github.com/brandonros/vanity-miner-rs/issues/38) | **Unconfirmed**; pointer-offset wording alone does not establish #76 ownership. | No demonstrated correction for this retained input. | No confirmed fix. | **Split-bundle translation:** LLVM 7 fails at line 306 after 0.350 s; LLVM 21 emits MSL in 0.409 s. Reduce the LLVM 7 offset types, then complete Apple compilation and CPU-verified GPU batches for that version. |
| [#39 — LLVM 7 P-256 public-key production](https://github.com/brandonros/vanity-miner-rs/issues/39) | **Unconfirmed**; reduce the CFG join before assigning a predicate/definedness owner. | No demonstrated correction for this retained input. | No confirmed fix. | **Split-bundle translation:** LLVM 7 reports `%rd18761` undefined at `$L__BB5_1` after 9.412 s; LLVM 21 emits MSL in 6.906 s. Preserve distinct version acceptance and complete bounded CPU-verified GPU batches. |


The two passing production rows are in the [status report](cumetal-status.md):
Shallenge completes in **2.03 s** and P-256 public key in **192.36 s**, each with
64 candidates and two CPU-verified outputs. These are bounded validation results,
not performance benchmarks or exhaustive mode coverage.

## Keep original fix evidence and changed inputs separate

| Scoped implementation | What the original evidence establishes | What the fresh run does not establish |
| --- | --- | --- |
| [#117](https://github.com/Lulzx/cuda-metal/pull/117) | Retained Bitcoin, Solana, Ethereum and P-256 public-key modules translate past trap expansion; old Bitcoin then waits in Metal compilation. | That changed self-test helpers/probes pass every earlier or later stage. |
| [#121](https://github.com/Lulzx/cuda-metal/pull/121) | Original P-256 signature self-test clears `%rs506` and reaches later type diagnostics. | That new `%p243` is the same defect or that removing the depth cap failed. |
| [#122](https://github.com/Lulzx/cuda-metal/pull/122) | Original RSA self-test clears `%rs2233`, emits 53.6 MB MSL, then hits the separately tracked #124 resource failure. | That new `range_multiple` `%rs907` reproduces the exact fixed case, or that #124 was fixed. |
| [#118](https://github.com/Lulzx/cuda-metal/issues/118), still without a PR | Historical unchanged RSA-PSS PTX has 63 mixed-vector pointer diagnostics, isolated by controlled probes. | That different, outlined PTX reaching Apple compilation fixes that original compiler gap. |

## Additional upstream work and boundaries

| Upstream issue | State at audit | Why it matters / what remains |
| --- | --- | --- |
| [#123 — helper global arguments](https://github.com/Lulzx/cuda-metal/issues/123) | **Local WIP; no commit/PR.** Public handoff records code and focused tests outside the pin. | Earlier outlined RSA input needs globals threaded through helper signatures/calls. Fresh P-256/RSA-PSS diagnostics add relevant investigation leads, not evidence this WIP fixes either complete module. Full baseline gates and publication remain pending. |
| [#125 — helper-only registration globals](https://github.com/Lulzx/cuda-metal/issues/125) | **No fix PR.** | Separate metadata traversal defect: helper-only globals miss initialization/persistence. The #123 small GPU fixture retains a direct entry reference to bypass it. Fresh source-compile errors do not test this runtime behavior. |
| [#124 — Metal allocation failure](https://github.com/Lulzx/cuda-metal/issues/124) | **Research; no demonstrated fix.** | Original monolithic and outlined historical artifacts retain separate results. Current RSA self-tests stop earlier in translation; that is neither an allocation-failure reproduction nor a repair. |
| [#127 — constant byte-shuffle specialization](https://github.com/Lulzx/cuda-metal/issues/127) | **Optimization not implemented; no PR.** | First concrete experiment for #124: immediate `prmt.b32` currently creates 66 values per shuffle. Its sequences occupy 35.55% of the retained RSA entry's source bytes; no speedup or crash correction has been demonstrated. |
| [#115 / tracing PR #128](https://github.com/Lulzx/cuda-metal/pull/128) | **Diagnostic implementation proposed; outside pin.** Performance remains unresolved. | Adds opt-in compilation spans; an end record means scope exit, including failure. New tests pass, but wider Release/selected Debug failures are documented. It neither supplies #126 nor fixes #124. |
| [#126 — explicit function preparation](https://github.com/Lulzx/cuda-metal/issues/126) | **Missing APIs; no fix PR.** | Add `cuFuncLoad`/`cuFuncIsLoaded` with shared preparation, readiness, errors and lifecycle tests. Preparation success remains separate from GPU results and hidden-buffer correctness. |

## Serial priority: LLVM 21 Solana, Bitcoin and Ethereum production

The [#76 implementation plan](cumetal-76-plan.md) specifies the coordinated
result-type, normalized-definition and join/loop changes, plus the tests required
to close that compiler defect. Full production GPU acceptance remains a separate
gate for each downstream issue.

For the new LLVM 7 inputs, Bitcoin and Ethereum can proceed directly to Apple
compilation and GPU acceptance; their translation successes are recorded in the
split-bundle comparison above. They do not establish the LLVM 21 workloads pass.

1. **Upstream [#76](https://github.com/Lulzx/cuda-metal/issues/76):** infer correct
   conversion result types before SSA, preserve instruction-specific types through
   CFG copies/rewrites, and resolve incoming join/loop types. Replay all three
   retained production PTX inputs after each #76 change; retain focused compiler
   regression tests. No published PR implements the remaining shared correction.
2. **Production [#23 — Solana](https://github.com/brandonros/vanity-miner-rs/issues/23)
   → [#24 — Bitcoin](https://github.com/brandonros/vanity-miner-rs/issues/24)
   → [#25 — Ethereum](https://github.com/brandonros/vanity-miner-rs/issues/25):**
   finish each workload through Metal compilation, GPU completion and the
   acceptance checks below. Insert each newly demonstrated production blocker
   when it is found, before proceeding to the next workload. The complete upstream
   issue sequence beyond #76 is unknown until these inputs reach later stages.
3. **Self-tests [#29 — Solana](https://github.com/brandonros/vanity-miner-rs/issues/29)
   → [#30 — Bitcoin](https://github.com/brandonros/vanity-miner-rs/issues/30)
   → [#31 — Ethereum](https://github.com/brandonros/vanity-miner-rs/issues/31):**
   complete all 79, 39 and 8 checks respectively as additional correctness
   coverage. These self-test passes are not established prerequisites for the
   production passes. Replaying #31 alongside #76 can test the suspected overlap;
   the shared cause remains unconfirmed.

**Production acceptance:** empty address patterns are insufficient because these
three kernels return only the private key for a matching candidate. Require
selective nonempty prefix/suffix fixtures with known matching and nonmatching
candidates, CPU-verified match/error counts and returned winners, and independent
CPU reconstruction of each winner's address. Check unchanged inputs and buffer
guards across at least two batches to exercise reuse. Record the selected source,
paired compiler/runtime and PTX identities with each result. The existing empty-
pattern measurements do not establish this acceptance gate.

No RSA, P-256 or Shallenge issue is currently a demonstrated dependency for these
three production workloads.

## Shared fixes and next assignments

- **Confirmed shared work:** production #23–25 share #76. Coordinate its importer
  result-type edits with #118's per-lane pointer work.
- **Research before assigning fresh duplicates:** new undefined-value errors in
  Solana, Bitcoin, Shallenge, P-256 signatures and RSA modulus need separate small
  reductions. Existing CFG/unused-bit owners are leads, not five confirmed copies
  of one defect.
- **Helper-global work:** keep #123 and #125 coordinated, while retaining the
  distinction between source generation, metadata discovery and GPU binding.
  Independently reduce the new Metal pointer-cast errors.
- **Resource investigations:** keep RSA-PSS library compilation (#115), historical
  RSA self-test allocation failure (#124), historical RSA GPU completion (#119),
  and fresh RSA pipeline-creation timeout separate until shared causes are proved.
- **One integration owner:** control the pin, compiler/runtime pairing, PTX
  identities and serialized GPU measurements. A proposed patch must state which
  exact reproducer clears and which next stage remains.

Fresh local evidence is retained in
`.cumetal-artifacts/validation-actions-35044328837-20260916T013211Z/`:
`results.json`, `metadata.json`, `cpu-baseline.json`,
`fresh-self-test-triage.json`, `ptx-structural-comparison.json`, and command logs.
These are local evidence paths, not public attachments. The detailed reports
link above record coverage and interpretation limits. Historical research below
remains attributed to its original source and inputs.

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
