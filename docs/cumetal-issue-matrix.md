# CuMetal issue ownership and fix status

Updated **2026-09-15**. This maps all **13 unresolved vanity-miner issues** to
their upstream CuMetal work. An implemented compiler fix does not establish that
the complete downstream workload passes.

The last complete [16-row validation](cumetal-status.md) tested CuMetal
`e5acf8cc0c658142c704ee80e749f5180911fff4`: **3 true / 13 false**. Targeted
research below tested `92a9b8f4de230199eac617fc57feaf7c0849cf01` (PR #114).
At the final lock check on **2026-09-15, 20:47 UTC**, the lock had advanced again
to `7d12f120a6b80a9956588de2b974db5747b35f57` (PR #117). Neither lock advance
constitutes a new complete local validation run. PR #117's separately reported
results are attributed explicitly below.

## What the statuses mean

**PR publication, implementation coverage, and complete GPU validation are
separate facts.** The earlier labels “required fix not implemented” and “partial”
did not tell you whether anyone had started coding.

| Explicit status | Meaning | What it does not establish |
| --- | --- | --- |
| **No published fix PR** | No upstream patch implementing this remaining case was identified. | That nobody has started. Research or local code may already exist. |
| **Active local implementation; no PR yet** | Work is underway, supported by an agent's report or inspected changes. | That the patch is complete, reviewed, or tested successfully. |
| **Earlier PR covers narrower cases** | The earlier patch implements a real correction, while this newly identified case is outside its coverage. | That the earlier work did nothing, or that the remaining case already has a follow-up patch. |
| **Fix proposed in a PR** | A published implementation exists. Its exact test evidence determines what it establishes. | That merging the PR makes every dependent workload pass. |
| **Identified blocker cleared; later blocker remains** | The relevant artifact passes that stage and reaches another failure. | That the first fix is incomplete merely because a different defect remains. |
| **Complete workload verified** | The unchanged workload executes and passes its required GPU/CPU checks on an identified build. | Exhaustive correctness outside that recorded coverage. |

“No published fix PR” is an audit of public work, not an assertion that no local
branch exists. “Not started” should be used only when the assigned owner confirms
that. None of the 13 downstream workloads has a complete passing result in the
recorded evidence.

## Matrix

**GitHub checked 2026-09-15, 21:12 UTC:** 69 issues and 51 PRs, including open and
closed entries. The newest published PR was #117. Agent activity below is
explicitly attributed to the user's report; no implementation work was launched
by this status audit.

| Vanity-miner issue | CuMetal owner | Published PR coverage | Work state / remaining gap | Downstream completion still required |
| --- | --- | --- | --- | --- |
| [#23 — Solana production](https://github.com/brandonros/vanity-miner-rs/issues/23) | [#76 — conversion/join types](https://github.com/Lulzx/cuda-metal/issues/76) | **No PR for the remaining conversion-destination fix.** Related #50/#56/#75 cover narrower cases. | Diagnosed; coding start for this remaining case is not established. #76 already includes the expanded acceptance. | Correct destination inference before SSA, translate unchanged PTX, and complete CPU-verified GPU batches. |
| [#24 — Bitcoin production](https://github.com/brandonros/vanity-miner-rs/issues/24) | [#76](https://github.com/Lulzx/cuda-metal/issues/76) | **No PR for this remaining gap.** Same shared work as #23. | One upstream implementation task, with separate downstream validation. | Retest this exact artifact and complete verified GPU batches. |
| [#25 — Ethereum production](https://github.com/brandonros/vanity-miner-rs/issues/25) | [#76](https://github.com/Lulzx/cuda-metal/issues/76) | **No PR for this remaining gap.** Same shared work as #23. | One upstream implementation task, with separate downstream validation. | Retest this exact artifact and complete verified GPU batches. |
| [#26 — P-256 signature production](https://github.com/brandonros/vanity-miner-rs/issues/26) | [#83 — predicate threading](https://github.com/Lulzx/cuda-metal/issues/83) | [PR #85](https://github.com/Lulzx/cuda-metal/pull/85) covers original cases; **no published empty-label follow-up**. | Expanded scope remains uncovered. Coordinate with the active CFG owner; do not assume its depth-cap change also fixes consecutive labels. | Preserve predicate facts across empty label blocks and validate both search sources. |
| [#27 — RSA modulus production](https://github.com/brandonros/vanity-miner-rs/issues/27) | [#119 — generation completion](https://github.com/Lulzx/cuda-metal/issues/119) | **No fix PR.** | Research established the active generation stage; the GPU wait's cause and corrective patch remain unproved. | Diagnose the wait, then complete the four-stage pipeline and verified cycles. |
| [#28 — RSA-PSS production](https://github.com/brandonros/vanity-miner-rs/issues/28) | Earlier [#96](https://github.com/Lulzx/cuda-metal/issues/96)/[#109](https://github.com/Lulzx/cuda-metal/issues/109)/[#111](https://github.com/Lulzx/cuda-metal/issues/111) → [#115 — Metal compilation time](https://github.com/Lulzx/cuda-metal/issues/115) | #108/#110/#112 clear earlier translation blockers. **No fix PR for #115.** | Earlier scoped fixes work on the tested full PTX; the later compilation problem needs research and a correction. | Complete Metal compilation, GPU execution and CPU verification for salt and message search. |
| [#29 — Solana self-tests](https://github.com/brandonros/vanity-miner-rs/issues/29) | [#46](https://github.com/Lulzx/cuda-metal/issues/46), [#116 — trap-call expansion](https://github.com/Lulzx/cuda-metal/issues/116) | Earlier #62; current [draft PR #117](https://github.com/Lulzx/cuda-metal/pull/117). | Published implementation. PR #117 reports full PTX translation; full GPU checks are not established. | Validate the proposed stack through all 72 checks. |
| [#30 — Bitcoin self-tests](https://github.com/brandonros/vanity-miner-rs/issues/30) | [#113 — pointer fields](https://github.com/Lulzx/cuda-metal/issues/113) → [#116](https://github.com/Lulzx/cuda-metal/issues/116) | [#114](https://github.com/Lulzx/cuda-metal/pull/114) clears pointer errors; [#117](https://github.com/Lulzx/cuda-metal/pull/117) clears expansion. **No fix PR for the subsequent Metal compilation wait.** | PR #117 reports full PTX translation and a compilation wait exceeding ten minutes. This later problem needs diagnosis/ownership; shared symptoms do not prove it is #115's cause. | Complete compilation and all 33 checks; assign the later failure explicitly before treating the workload as resolved. |
| [#31 — Ethereum self-tests](https://github.com/brandonros/vanity-miner-rs/issues/31) | [#46](https://github.com/Lulzx/cuda-metal/issues/46), [#116](https://github.com/Lulzx/cuda-metal/issues/116) | [Draft PR #117](https://github.com/Lulzx/cuda-metal/pull/117). | Published implementation; full PTX translation reported, GPU checks pending. | Validate all five checks. |
| [#32 — P-256 public-key self-tests](https://github.com/brandonros/vanity-miner-rs/issues/32) | [#46](https://github.com/Lulzx/cuda-metal/issues/46), [#116](https://github.com/Lulzx/cuda-metal/issues/116) | [Draft PR #117](https://github.com/Lulzx/cuda-metal/pull/117). | Published implementation; full PTX translation reported, GPU checks pending. | Validate all nine checks. |
| [#33 — P-256 signature self-tests](https://github.com/brandonros/vanity-miner-rs/issues/33) | Broad [#83](https://github.com/Lulzx/cuda-metal/issues/83); specific [#120 — eight-block guard limit](https://github.com/Lulzx/cuda-metal/issues/120) | Earlier [#85](https://github.com/Lulzx/cuda-metal/pull/85); **no published #120 PR yet**. | **Active local implementation by the user's other agent.** New #120 documents the depth-cap case. Do not assign a second competing CFG patch. | Verify the final patch against unchanged PTX, retain genuine-undefined-path rejection, and execute all ten checks. |
| [#34 — RSA modulus self-tests](https://github.com/brandonros/vanity-miner-rs/issues/34) | Earlier [cuda-metal #35](https://github.com/Lulzx/cuda-metal/issues/35); active shared investigation [#120](https://github.com/Lulzx/cuda-metal/issues/120) | Earlier [#57](https://github.com/Lulzx/cuda-metal/pull/57); **no published follow-up PR yet**. | Other agent reports the same depth cap clears the undefined-register stage. Confirm on the final unchanged artifact before treating the older masked-payload proposal as a separate required patch. | Resolve any subsequent type failures and execute all 12 checks. |
| [#35 — RSA-PSS self-tests](https://github.com/brandonros/vanity-miner-rs/issues/35) | [#118 — mixed local-vector pointers](https://github.com/Lulzx/cuda-metal/issues/118) → related [#116](https://github.com/Lulzx/cuda-metal/issues/116) | **No fix PR for #118.** #108/#114 implement different pointer cases; #117 addresses subsequent trap work. | First defect isolated and ready for implementation. A diagnostic scalar reload removes 63 errors and exposes a later expansion failure; test the combined fixes rather than assuming #117 covers that case. | Translate unchanged full PTX, execute ten enabled checks and report the intentional skip. |

## Scope changes and agent assignment

- **Already expanded:** #76, #83 and upstream #35 explicitly include remaining
  cases beyond earlier PRs. Their presence does not mean those earlier PRs
  promised or implemented the added cases.
- **Separate later blockers:** #115, #118 and #119 have their own owners. #120
  now provides a more specific owner for active guard-depth work. The Bitcoin
  Metal-compilation wait still needs a confirmed owner or a justified extension
  of an existing investigation.
- **Resolve overlap before assigning:** #46/#116 describe related trap work now
  addressed by PR #117. #83/upstream #35/#120 may overlap on the active CFG fix.
  Keep that work with the existing agent until its exact artifact results define
  what remains; do not launch one competing implementation per issue number.

For a future swarm, assign one owner each to conversion types (#76), mixed-vector
pointer recovery (#118), RSA-PSS compilation research (#115), RSA generation
research (#119), and the existing CFG work (#120 and related cases). A single
integration owner should manage the consumer pin, combined builds and serialized
GPU validation. Workers should exchange exact commits, PTX hashes, confirmed
scope, next blockers and test outcomes. The two compilation waits can be compared
without prematurely declaring one root cause.

PR #117's results in this matrix are its author's reports, outside the complete
`e5acf8` run and targeted `92a9b8` research. The current consumer pin contains
#117, but pinning alone does not supply a complete passing validation result.

## Targeted research: RSA modulus production (#27)

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

## Targeted research: RSA-PSS self-tests (#35)

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
The later PR #117 must be tested against this RSA-PSS case after pointer recovery
is corrected. [Upstream #118](https://github.com/Lulzx/cuda-metal/issues/118)
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
