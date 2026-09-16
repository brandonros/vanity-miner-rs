# CuMetal issue consolidation audit

Initial snapshot audited **2026-09-16 UTC**: 68 open issues authored by `brandonros` in
`Lulzx/cuda-metal`; 64 in the previously reported broad compiler-related scope.
Exclude #24 (runtime provenance), #91 (Nix environment), #119 (GPU-completion
investigation), and #126 (function-preparation APIs). The 64 still include
performance investigations and #125's compiler/registration boundary; this is
not a count of 64 established compiler defects.

The initial audit used issue bodies, comments and PR scopes, without compiler,
Metal or GPU runs. The subsequent [executed follow-up](cumetal-consolidation-followup.md)
published the consolidation and adds CPU translation/expression evidence.
Results attributed to existing issues/PRs remain historical
or author-reported. PR closing references identify proposed ownership, not proof
that every acceptance test passes. The captured issues and PRs were open at the
initial audit. The follow-up closed #46 as superseded and updated the other
identified scopes; no fix PR was merged.

## Decisions and initial-snapshot counts

- **0 exact duplicates established** by the four-part test: failing input/stage,
  demonstrated cause, correction/regressions, and completion criteria. This is
  not proof that no undiscovered common root cause exists.
- **1 full-tracker consolidation completed:** #46 was closed as superseded after transferring
  its acceptance into #116. The implementation plan is superseded, not an identical
  original report. This leaves **63 of the audited trackers open**, not 63 established bugs.
- **1 partial overlap, corrected publicly:** #83's historical self-test depth
  failure is assigned to #120/#121; #83 retains empty-label handling and its
  original regressions.
- **2 unresolved resource attributions:** #115 and #124. New measurements
  identify shared PRMT expansion, but not one proven timeout/crash cause.
- **#80 now has an isolated missing inference edge:** generic-to-global helper
  pointer recovery. Retain it as a separate extension; no fix is implemented.
- **59 other separate scopes:** retain their independent acceptance. Many are
  related or dependent, rather than unrelated defects.

Implementation is a separate axis: **51** issues have an open PR explicitly
claiming to fix/close them; #46 has the shared replacement #117; #83 has partial
implementation #85; **11** have no proposed fix PR identified:
#76, #77, #80, #115, #118, #123, #124, #125, #127, #129, #130.
For #115, diagnostic PR #128 does not count as a performance fix. #123 reports
local WIP, which does not establish a published implementation. Open draft PRs
and inclusion in the consumer's fork pin are different from upstream integration.

## Reconcile #46 and #116

[The existing #46 follow-up](https://github.com/Lulzx/cuda-metal/issues/46#issuecomment-5688538764)
already says [#117](https://github.com/Lulzx/cuda-metal/pull/117) supersedes the
finite-helper expansion plan. The **initial body** still asked for that old implementation,
while [#116](https://github.com/Lulzx/cuda-metal/issues/116) owns the replacement.
That disagreement created the apparent extra unfinished fix; the follow-up
replaced the stale current-body status and preserved it as history.

Use **#116 as the canonical compiler issue** and **#117 as its implementation**.
The following acceptance reconciliation has been published in the GitHub
bodies. The transferred gates are not all claimed complete.

| Old requirement | Reconciled requirement / owner |
| --- | --- |
| Prove constant-trip loops so selected helpers avoid expansion. | Preserve reusable finite helpers as calls. Do not require the obsolete acyclic/finite-loop classifier: #117 removes expansion. Removing cancellation polls through a termination proof would be a separate optimization. |
| Expand trapping or unproven helpers into the kernel. | Retain supported trap-capable helpers with shared hidden status, propagate errors through nested calls, and poll cancellation in cyclic dispatchers and after calls before consuming returns. |
| Preserve helper return values and local mutations. | Keep exact scalar/aggregate returns, private-pointer writes and memory ordering. Include a small counted-loop helper called repeatedly/nested, in guarded and ordinary forms. |
| Trap safety and progress. | Test a trapping lane against a spinning sibling through nested helpers, both divergent orders, with bounded completion and correct status. Cancellation responsiveness is not proof that an arbitrary program terminates without a trap. |
| Reject unsupported call graphs; failed legalization must not mutate IR. | Retain negative coverage for barriers, collectives, printf, unresolved/indirect calls and unsupported recursion/call forms. Check input IR is unchanged on rejection. |
| Expansion budgets and growth diagnostics. | Replace obsolete expansion counters with retained-helper/source-size checks. Preserve #117's 1,025-call / one-helper / under-1-MB regression and conservative analysis/resource limits; report remaining rejection causes. |
| Full Solana/Ethereum/P-256 self-tests must pass. | Preserve those workload gates in vanity-miner #29/#31/#32. Upstream closure requires scoped semantic tests and replay of the exact retained PTX past this blocker, with later failures separately owned; source emission alone does not satisfy downstream GPU acceptance. |

The last row separates compiler-fix closure from full workload closure. This
transfer was recorded in #116 before retiring #46; unchecked workload acceptance
remains in the downstream trackers.

### Retained artifact acceptance

| Original input | PTX SHA-256 | Existing evidence and remaining gate |
| --- | --- | --- |
| Solana, historical 72 checks | `e9ef053adf5b4f70d3607dcbae3890ae5886cfe59303b8f436be012ee47bf9e2` | #117/#46 follow-up report MSL emission in 11.21 s; full GPU acceptance remains vanity-miner #29. |
| Ethereum, historical 5 checks | `b52aeb7e75ef61a25166658644c6cfbc9fc6ef1e7454176677d745503617872d` | Reported emission in 15.60 s; full GPU acceptance remains vanity-miner #31. |
| P-256 public key, historical 9 checks | `a83bdc53edb56127a79bbbe0f638ef7d3a0d9125c419a775e40477e8b8b5ec19` | Reported emission in 11.08 s; full GPU acceptance remains vanity-miner #32. |
| Bitcoin, historical 33 checks | `1f16b59a3719a862d7aa56636f5392ca45837d1ecb1bd5778e31554d766d18a8` | #117 reports 19.77 MB MSL and a later Apple compilation wait; full GPU acceptance remains vanity-miner #30. |

For the merge candidate, retain exact compiler/runtime and input identities,
focused numerical/guard tests, and the full Release/Debug failure-name comparison.
#117 reports focused tests passing and baseline-matching failures (58/342 Release,
58/344 Debug), **not an entirely passing suite**. Extra acceptance proposed above
must be matched to named tests/results before declaring it fulfilled.

RSA-PSS remains conditional on [#118](https://github.com/Lulzx/cuda-metal/issues/118):
the scalar-reload diagnostic copy reached expansion failure, but that modified
input is not an unchanged full-module pass. Retest the original after pointer
recovery. New split self-test modules have different hashes/check counts; their
later errors do not reopen this scoped expansion defect without a reproducer.

### Published GitHub reconciliation

For **#46**, the stale status/checklist was replaced with this summary and the
old report retained in historical details:

> Superseded by #116 / PR #117. The remaining implementation is reusable guarded
> device helpers with shared trap status and cancellation, not a new finite-loop
> classifier for the removed expansion pass. Carry the original Solana, Ethereum
> and P-256 artifact identities and helper value/effect/cancellation regressions
> into #116's acceptance. Full GPU workload gates remain vanity-miner #29, #31
> and #32 (Bitcoin separately #30). Reported MSL emission clears the old expansion
> stage only. This tracker is retired as superseded after recording the acceptance
> transfer; #116 stays open through review and integration.

For **#116**, the reconciled table and artifact ledger above were appended,
explicitly linking #46. For **#117**, the update adds `Refs #46; supersedes its expansion-based plan` and
records outstanding review/verification checkboxes for retained gates. Its current closing
claim is **`Fixes #116` only**, not #46. Closing #46 administratively after transfer
must not be presented as the compiler fix merging.

## The eight PRs do explicitly name their multiple issues

Every PR below is open and draft in the snapshot. Each linked body contains a
separate `Fixes #N` phrase for every listed issue. They cover **24 unique issue
IDs**, and no additional closing-reference text is needed merely to associate
them. Separate regression/acceptance cases should remain visible during review.

| PR | Explicit closing references | Why retain separate acceptance |
| --- | --- | --- |
| [#50](https://github.com/Lulzx/cuda-metal/pull/50) | [#26](https://github.com/Lulzx/cuda-metal/issues/26), [#32](https://github.com/Lulzx/cuda-metal/issues/32), [#44](https://github.com/Lulzx/cuda-metal/issues/44), [#47](https://github.com/Lulzx/cuda-metal/issues/47), [#48](https://github.com/Lulzx/cuda-metal/issues/48) | Memory store/load widths, unsigned literal widening, parameter stores, conversion source width. |
| [#52](https://github.com/Lulzx/cuda-metal/pull/52) | [#25](https://github.com/Lulzx/cuda-metal/issues/25), [#30](https://github.com/Lulzx/cuda-metal/issues/30) | Selected-entry validation and immutable pointer relocation are independent contracts. |
| [#54](https://github.com/Lulzx/cuda-metal/pull/54) | [#27](https://github.com/Lulzx/cuda-metal/issues/27), [#28](https://github.com/Lulzx/cuda-metal/issues/28), [#29](https://github.com/Lulzx/cuda-metal/issues/29), [#33](https://github.com/Lulzx/cuda-metal/issues/33), [#34](https://github.com/Lulzx/cuda-metal/issues/34) | Literal vector lanes, funnel shifts, PRMT, BFI, halfword packing. |
| [#55](https://github.com/Lulzx/cuda-metal/pull/55) | [#36](https://github.com/Lulzx/cuda-metal/issues/36), [#38](https://github.com/Lulzx/cuda-metal/issues/38) | Min/max typing and exact high-half multiplication. |
| [#56](https://github.com/Lulzx/cuda-metal/pull/56) | [#37](https://github.com/Lulzx/cuda-metal/issues/37), [#39](https://github.com/Lulzx/cuda-metal/issues/39), [#45](https://github.com/Lulzx/cuda-metal/issues/45) | Unused pointer casts, helper-pointer inference, pointer-minus-offset lowering. |
| [#58](https://github.com/Lulzx/cuda-metal/pull/58) | [#40](https://github.com/Lulzx/cuda-metal/issues/40), [#41](https://github.com/Lulzx/cuda-metal/issues/41), [#42](https://github.com/Lulzx/cuda-metal/issues/42) | Scalar tail calls, vector parameter lanes, safe local-buffer reuse. |
| [#70](https://github.com/Lulzx/cuda-metal/pull/70) | [#51](https://github.com/Lulzx/cuda-metal/issues/51), [#66](https://github.com/Lulzx/cuda-metal/issues/66) | Explicit discarded half plus the named-but-unread destination extension. |
| [#71](https://github.com/Lulzx/cuda-metal/pull/71) | [#60](https://github.com/Lulzx/cuda-metal/issues/60), [#65](https://github.com/Lulzx/cuda-metal/issues/65) | Parameter classification plus sidecar agreement with emitted ABI. |

## All 64 issue decisions

“Closing PR” means a published explicit closing claim, not a fresh verification.
“Retain” means acceptance is not fully subsumed by another issue; it does not
prove an independent root cause. Linked issue bodies contain the reproducer,
stage, constraints and original acceptance. Latest comments take precedence over
stale body summaries where explicitly identified above.

| Issue and scoped failure | Recommendation | Proposed implementation | Evidence for the boundary |
| --- | --- | --- | --- |
| [#23](https://github.com/Lulzx/cuda-metal/issues/23) — MSL registration sidecar emission | Retain separate scope | [#49](https://github.com/Lulzx/cuda-metal/pull/49) (closing claim) | Missing metadata output; #65 instead corrects the ABI recorded in an existing sidecar. |
| [#25](https://github.com/Lulzx/cuda-metal/issues/25) — Selected-entry reachability before validation | Retain separate scope | [#52](https://github.com/Lulzx/cuda-metal/pull/52) (closing claim) | Unreachable unsupported declarations/instructions must not reject the selected entry. #30 additionally resolves reachable initializer relocations. |
| [#26](https://github.com/Lulzx/cuda-metal/issues/26) — Narrow ordinary-memory stores | Retain separate scope | [#50](https://github.com/Lulzx/cuda-metal/pull/50) (closing claim) | Truncate to the memory width and preserve neighboring bytes; #47 exercises parameter slots, a separate lowering path. |
| [#27](https://github.com/Lulzx/cuda-metal/issues/27) — Vector-store literal tuple lanes | Retain separate scope | [#54](https://github.com/Lulzx/cuda-metal/pull/54) (closing claim) | Parse literal and register components without dropping a lane; independent of the arithmetic opcodes in the same PR. |
| [#28](https://github.com/Lulzx/cuda-metal/issues/28) — Funnel-shift wrap semantics | Retain separate scope | [#54](https://github.com/Lulzx/cuda-metal/pull/54) (closing claim) | Exact 32-bit shift/count behavior; neither PRMT (#29) nor bit-field insertion (#33) covers this contract. |
| [#29](https://github.com/Lulzx/cuda-metal/issues/29) — PRMT correctness | Retain separate scope | [#54](https://github.com/Lulzx/cuda-metal/pull/54) (closing claim) | Implement byte selection/sign-copy semantics. #127 is a later constant-selector optimization, not missing opcode support. |
| [#30](https://github.com/Lulzx/cuda-metal/issues/30) — Immutable pointer-table relocations | Retain separate scope | [#52](https://github.com/Lulzx/cuda-metal/pull/52) (closing claim) | Resolve supported symbolic pointers and reject ambiguous/partial relocations; distinct from unreachable-code validation (#25) and helper read-only proof (#89). |
| [#31](https://github.com/Lulzx/cuda-metal/issues/31) — Multiline call parsing | Retain separate scope | [#53](https://github.com/Lulzx/cuda-metal/pull/53) (closing claim) | Recover complete call statements before lowering; unrelated to call cancellation or recursion semantics. |
| [#32](https://github.com/Lulzx/cuda-metal/issues/32) — Narrow memory-load extension | Retain separate scope | [#50](https://github.com/Lulzx/cuda-metal/pull/50) (closing claim) | Preserve load width and sign extension; #26 concerns writes, #48 conversion instructions. |
| [#33](https://github.com/Lulzx/cuda-metal/issues/33) — Bit-field insertion | Retain separate scope | [#54](https://github.com/Lulzx/cuda-metal/pull/54) (closing claim) | Implement bounded BFI semantics; distinct opcode and boundary tests from #28/#29. |
| [#34](https://github.com/Lulzx/cuda-metal/issues/34) — 32-bit tuple pack/unpack | Retain separate scope | [#54](https://github.com/Lulzx/cuda-metal/pull/54) (closing claim) | Correct two-halfword ordering/value behavior; #64 concerns 64-bit inferred widths, #51 undefined but unobserved bits. |
| [#35](https://github.com/Lulzx/cuda-metal/issues/35) — Masked optional-payload definedness | Retain separate scope | [#122](https://github.com/Lulzx/cuda-metal/pull/122) (closing claim) | #57 addressed earlier bounded cases; #122 addresses the retained masked-comparison case. #120's depth fix explicitly did not clear it. Changed-input failures require fresh attribution. |
| [#36](https://github.com/Lulzx/cuda-metal/issues/36) — Min/max operand types | Retain separate scope | [#55](https://github.com/Lulzx/cuda-metal/pull/55) (closing claim) | Resolve signedness and MSL overloads; #38 instead requires exact high-half multiplication. |
| [#37](https://github.com/Lulzx/cuda-metal/issues/37) — Unused narrow pointer conversions | Retain separate scope | [#56](https://github.com/Lulzx/cuda-metal/pull/56) (closing claim) | Remove only unobserved unsupported conversions; preserve rejection when observed. Not helper pointer inference (#39). |
| [#38](https://github.com/Lulzx/cuda-metal/issues/38) — 64-bit high-half multiplication | Retain separate scope | [#55](https://github.com/Lulzx/cuda-metal/pull/55) (closing claim) | Signed/unsigned mul.hi numeric semantics; independent of min/max (#36). |
| [#39](https://github.com/Lulzx/cuda-metal/issues/39) — Generic helper-pointer inference | Retain separate scope | [#56](https://github.com/Lulzx/cuda-metal/pull/56) (closing claim) | Generic-to-local helper recovery; the follow-up isolates #80 as missing generic-to-global recovery with a separate index parameter. |
| [#40](https://github.com/Lulzx/cuda-metal/issues/40) — Scalar tail-self-call elimination | Retain separate scope | [#58](https://github.com/Lulzx/cuda-metal/pull/58) (closing claim) | Preserve scalar arguments and aggregate returns when lowering supported recursion to a loop; #42 adds local-buffer lifetime/read constraints. |
| [#41](https://github.com/Lulzx/cuda-metal/issues/41) — Vector parameter transfers | Retain separate scope | [#58](https://github.com/Lulzx/cuda-metal/pull/58) (closing claim) | Preserve every argument/return lane even without recursion; independent regression from #40. |
| [#42](https://github.com/Lulzx/cuda-metal/issues/42) — Local-buffer tail-call reuse | Retain separate scope | [#58](https://github.com/Lulzx/cuda-metal/pull/58) (closing claim) | Requires a local-buffer read-all/escape proof beyond scalar tail calls (#40); retain both positive and rejection tests. |
| [#43](https://github.com/Lulzx/cuda-metal/issues/43) — Kernel trap reporting/cancellation | Retain separate scope | [#62](https://github.com/Lulzx/cuda-metal/pull/62) (closing claim) | Kernel status, stream errors and sibling cancellation remain a contract separate from retaining device calls (#116). |
| [#44](https://github.com/Lulzx/cuda-metal/issues/44) — Negative-spelled unsigned literals | Retain separate scope | [#50](https://github.com/Lulzx/cuda-metal/pull/50) (closing claim) | Apply source-width unsigned interpretation before widening; narrower source-literal contract than conversion destination/join inference (#76). |
| [#45](https://github.com/Lulzx/cuda-metal/issues/45) — Pointer-minus-integer lowering | Retain separate scope | [#56](https://github.com/Lulzx/cuda-metal/pull/56) (closing claim) | Correctly typed pointer minus byte offset; explicitly excludes integer-minus-pointer. #129 introduces a proved same-base cancellation case outside that scope. |
| [#46](https://github.com/Lulzx/cuda-metal/issues/46) — Old finite-helper expansion plan | Closed as superseded into #116 | Shared [#117](https://github.com/Lulzx/cuda-metal/pull/117) | Superseded by #116/#117's reusable guarded helpers. Transfer artifact and semantic acceptance; retire the finite-loop-classifier implementation requirement. |
| [#47](https://github.com/Lulzx/cuda-metal/issues/47) — Narrow parameter stores | Retain separate scope | [#50](https://github.com/Lulzx/cuda-metal/pull/50) (closing claim) | Truncate call argument/return slots; #26's ordinary-memory store tests do not establish parameter ABI correctness. |
| [#48](https://github.com/Lulzx/cuda-metal/issues/48) — Conversion source-width semantics | Retain separate scope | [#50](https://github.com/Lulzx/cuda-metal/pull/50) (closing claim) | Read the source with the specified width/sign; #76 adds destination inference and normalized/join definitions. |
| [#51](https://github.com/Lulzx/cuda-metal/issues/51) — Explicitly discarded packed half | Retain separate scope | [#70](https://github.com/Lulzx/cuda-metal/pull/70) (closing claim) | Bounded pack/extract proof with an explicit discard. #66 adds whole-function proof that a named extraction result is unread. |
| [#59](https://github.com/Lulzx/cuda-metal/issues/59) — Store-address provenance | Retain separate scope | [#68](https://github.com/Lulzx/cuda-metal/pull/68) (closing claim) | A store reads its address; it must not redefine the address register's parameter provenance. Distinct from #60 fallback classification and #61 operand order. |
| [#60](https://github.com/Lulzx/cuda-metal/issues/60) — 64-bit parameter classification | Retain separate scope | [#71](https://github.com/Lulzx/cuda-metal/pull/71) (closing claim) | Width alone is not pointer evidence. #65 separately ensures the emitted ABI is faithfully exported to the sidecar. |
| [#61](https://github.com/Lulzx/cuda-metal/issues/61) — Commuted pointer addition | Retain separate scope | [#72](https://github.com/Lulzx/cuda-metal/pull/72) (closing claim) | Choose the pointer operand from provenance, not operand position; #59/#60 can be correct while this still fails. |
| [#64](https://github.com/Lulzx/cuda-metal/issues/64) — 64-bit tuple inferred width | Retain separate scope | [#69](https://github.com/Lulzx/cuda-metal/pull/69) (closing claim) | Do not corrupt packed loop state through a 32-bit type seed; distinct from #34's halfword ordering and #51's unobserved bits. |
| [#65](https://github.com/Lulzx/cuda-metal/issues/65) — Typed-emitter sidecar ABI | Retain separate scope | [#71](https://github.com/Lulzx/cuda-metal/pull/71) (closing claim) | Describe actual emitted pointer/scalar arguments; changing parser inference (#60) alone does not guarantee matching registration metadata. |
| [#66](https://github.com/Lulzx/cuda-metal/issues/66) — Unread named extraction destination | Retain separate scope | [#70](https://github.com/Lulzx/cuda-metal/pull/70) (closing claim) | Extension of #51 requiring a named destination to have no reads anywhere in the function; not covered by an explicit discard fixture. |
| [#67](https://github.com/Lulzx/cuda-metal/issues/67) — Promoted-global address spaces | Retain separate scope | [#74](https://github.com/Lulzx/cuda-metal/pull/74) (closing claim) | Preserve proven constant storage through conversions/helper use; separate from deciding whether storage is mutable (#77/#78). |
| [#73](https://github.com/Lulzx/cuda-metal/issues/73) — Per-instruction definition types | Retain separate scope | [#75](https://github.com/Lulzx/cuda-metal/pull/75) (closing claim) | Straight-line scalar-to-pointer register reuse; PR #75 explicitly leaves joins/normalized definitions to #76. |
| [#76](https://github.com/Lulzx/cuda-metal/issues/76) — Conversion destination and join types | Retain separate scope | No fix PR identified | Still needs destination inference, synthesized definitions, scalar/pointer joins and loop regressions; #48/#73 are narrower implemented prerequisites. |
| [#77](https://github.com/Lulzx/cuda-metal/issues/77) — Mutation through helper arguments | Retain separate scope | No fix PR identified | Interprocedural write/escape classification for private globals; #78 covers same-function aliases only. #123 assumes mutable classification already succeeded. |
| [#78](https://github.com/Lulzx/cuda-metal/issues/78) — Mutation through local aliases | Retain separate scope | [#79](https://github.com/Lulzx/cuda-metal/pull/79) (closing claim) | Same-function alias writes must prevent constant promotion; retaining #77's helper-mutation acceptance is necessary. |
| [#80](https://github.com/Lulzx/cuda-metal/issues/80) — Indexed generic helper-pointer conversion | Retain; missing inference edge isolated | No fix PR identified | Follow-up reproduces both originals and isolates recovery stopping at helper cvta.to.global with a separate index parameter. Pointer/move controls emit MSL; distinct from #39 local-helper and #76 join acceptance. |
| [#81](https://github.com/Lulzx/cuda-metal/issues/81) — Null pointer CFG joins | Retain separate scope | [#82](https://github.com/Lulzx/cuda-metal/pull/82) (closing claim) | Materialize a proven zero as a typed pointer on CFG edges; #109 handles already-pointer-typed nonzero immediates in selects. |
| [#83](https://github.com/Lulzx/cuda-metal/issues/83) — Constant predicates and empty-label paths | Reconcile partial overlap | Partial [#85](https://github.com/Lulzx/cuda-metal/pull/85) | Partial overlap: the historical self-test depth failure moved to #120/#121. Keep empty-label production acceptance and original #85 regressions here; do not retain the cleared depth defect as unimplemented work. |
| [#84](https://github.com/Lulzx/cuda-metal/issues/84) — Repeated unchanged predicates | Retain separate scope | [#86](https://github.com/Lulzx/cuda-metal/pull/86) (closing claim) | Retain guard identity across paths, including 16-bit comparisons; #93 recomputes a relational comparison, #130 transfers scalar sentinel facts. |
| [#87](https://github.com/Lulzx/cuda-metal/issues/87) — Redundant block arguments | Retain separate scope | [#88](https://github.com/Lulzx/cuda-metal/pull/88) (closing claim) | Fold already-valid identical incoming SSA values; does not establish missing type/definedness facts (#76/#35) or dominance-storage complexity (#94). |
| [#89](https://github.com/Lulzx/cuda-metal/issues/89) — Read-only helper use of relocated tables | Retain separate scope | [#90](https://github.com/Lulzx/cuda-metal/pull/90) (closing claim) | Prove passing an immutable table to a helper is read-only, extending #30. This is not #77's mutable helper case. |
| [#93](https://github.com/Lulzx/cuda-metal/issues/93) — Repeated unsigned variable comparisons | Retain separate scope | [#95](https://github.com/Lulzx/cuda-metal/pull/95) (closing claim) | Relate recomputed comparisons with unchanged operands; #84 preserves an existing predicate, #103 proves equal expressions in different registers. |
| [#94](https://github.com/Lulzx/cuda-metal/issues/94) — Dominator verification storage | Retain separate scope | [#97](https://github.com/Lulzx/cuda-metal/pull/97) (closing claim) | Eliminate quadratic host verifier storage without weakening dominance; different stage from Apple Metal resource investigations (#115/#124). |
| [#96](https://github.com/Lulzx/cuda-metal/issues/96) — Local pointer-table recovery | Retain separate scope | [#108](https://github.com/Lulzx/cuda-metal/pull/108) (closing claim) | Recover pointers stored/reloaded through local tables with offsets; #113 follows device-address demand through a scalar private-record field, #118 handles heterogeneous vector lanes. |
| [#98](https://github.com/Lulzx/cuda-metal/issues/98) — Unobserved retry copy cycles | Retain separate scope | [#100](https://github.com/Lulzx/cuda-metal/pull/100) (closing claim) | Remove dead payload copy round trips while preserving effects. #99 relates select/branch predicates, #101 preserves facts across calls. |
| [#99](https://github.com/Lulzx/cuda-metal/issues/99) — Complementary comparison predicates | Retain separate scope | [#102](https://github.com/Lulzx/cuda-metal/pull/102) (closing claim) | Prove eq/ne relationships feeding guarded self-selects; separate from copy liveness (#98) and repeated relational branch comparisons (#93). |
| [#101](https://github.com/Lulzx/cuda-metal/issues/101) — Caller facts across direct calls | Retain separate scope | [#104](https://github.com/Lulzx/cuda-metal/pull/104) (closing claim) | Preserve function-local predicate constants except actual return/clobber destinations; #103 handles expression equivalence, #106 analysis capacity. |
| [#103](https://github.com/Lulzx/cuda-metal/issues/103) — Equivalent integer expressions | Retain separate scope | [#105](https://github.com/Lulzx/cuda-metal/pull/105) (closing claim) | Relate repeated add/cvt values computed into different registers; not scalar zero-sentinel propagation (#130). |
| [#106](https://github.com/Lulzx/cuda-metal/issues/106) — Dead facts exhausting analysis budget | Retain separate scope | [#107](https://github.com/Lulzx/cuda-metal/pull/107) (closing claim) | Prune unused predicates before capacity fallback; neither call effects (#101) nor path depth (#120) is the demonstrated cause. |
| [#109](https://github.com/Lulzx/cuda-metal/issues/109) — Nonzero pointer immediates in MSL selects | Retain separate scope | [#110](https://github.com/Lulzx/cuda-metal/pull/110) (closing claim) | Importer already has pointer type; emitter must retain it. #81's zero CFG join case and #96's inference correction do not cover this. |
| [#111](https://github.com/Lulzx/cuda-metal/issues/111) — Parameter binding before CFG emission | Retain separate scope | [#112](https://github.com/Lulzx/cuda-metal/pull/112) (closing claim) | Dominating ld.param can appear later in source order; prebind values so MSL uses a declared identifier. Distinct from pointer inference and undefined PTX registers. |
| [#113](https://github.com/Lulzx/cuda-metal/issues/113) — Device pointers in private record fields | Retain separate scope | [#114](https://github.com/Lulzx/cuda-metal/pull/114) (closing claim) | Single-definition scalar load demanded as a device address; #96's table recovery and #118's mixed vector load require different proofs/tests. |
| [#115](https://github.com/Lulzx/cuda-metal/issues/115) — RSA-PSS Metal compilation latency | Attribution unresolved; retain | No fix PR identified | Root cause unresolved. #128 is instrumentation, not a performance fix. #124's allocation abort is a related investigation, not a demonstrated duplicate. |
| [#116](https://github.com/Lulzx/cuda-metal/issues/116) — Reusable trap-capable helpers | Retain separate scope | [#117](https://github.com/Lulzx/cuda-metal/pull/117) (closing claim) | Canonical owner for the #46 consolidation; shared status and cyclic-helper polling replace repeated expansion. Keep #43's kernel trap contract as a prerequisite. |
| [#118](https://github.com/Lulzx/cuda-metal/issues/118) — Mixed pointer/integer vector loads | Retain separate scope | No fix PR identified | Per-lane provenance for pointer/length reloads; homogeneous table recovery (#96) and scalar record loads (#113) do not cover it. |
| [#120](https://github.com/Lulzx/cuda-metal/issues/120) — Eight-block guard depth cutoff | Retain separate scope | [#121](https://github.com/Lulzx/cuda-metal/pull/121) (closing claim) | Nine-block fixture isolates the depth cap; removes the historical self-test subcase from #83, but not its empty-label failure or #106's fact capacity limit. |
| [#123](https://github.com/Lulzx/cuda-metal/issues/123) — Hidden mutable-global helper arguments | Retain separate scope | No fix PR identified | Thread already-promoted buffers through helper signatures/calls. #77/#78 classify storage; #125 discovers runtime registration data. |
| [#124](https://github.com/Lulzx/cuda-metal/issues/124) — Metal allocation failure | Attribution unresolved; retain | No fix PR identified | Root cause unresolved for the exact retained RSA-modulus source. #115 is latency on a different artifact; #127 is a proposed optimization experiment, not a proved repair. |
| [#125](https://github.com/Lulzx/cuda-metal/issues/125) — Helper-only global registration | Retain separate scope | No fix PR identified | Discover/initialize/persist globals referenced only by reachable helpers; depends on #123 but fixes metadata/runtime behavior rather than source signatures. |
| [#127](https://github.com/Lulzx/cuda-metal/issues/127) — Immediate PRMT specialization | Retain separate scope | No fix PR identified | Reduce generated IR/MSL for constant selectors while preserving #29's opcode semantics; success must be measured independently of #124's resource failure. |
| [#129](https://github.com/Lulzx/cuda-metal/issues/129) — Same-base address cancellation | Retain separate scope | No fix PR identified | LLVM 7 Shallenge needs a proof reducing address expressions to a scalar count; intentionally outside #45's pointer-minus-integer support and #76's conversion/join correction. |
| [#130](https://github.com/Lulzx/cuda-metal/issues/130) — Scalar zero-sentinel guards | Retain separate scope | No fix PR identified | LLVM 7 P-256 needs scalar move/zero-test facts; reducer has no empty labels, deep path, calls or budget pressure. Separate from #83/#120/#103. |

## Other overlaps that should not cause duplicate closures

- **#83 / #120:** [the existing scope correction](https://github.com/Lulzx/cuda-metal/issues/83#issuecomment-5688533710)
  assigns the old `%rs506` depth failure to #120/#121. Remove that obsolete
  “unimplemented” subcase from #83's status, but retain empty-label handling.
  Remaining cutoff code is a robustness test target, not proof of another
  currently reproduced failure.
- **#35 / #120:** [the recorded comparison](https://github.com/Lulzx/cuda-metal/issues/35#issuecomment-5688552951)
  explicitly distinguishes masked payload demand from the eight-block cutoff.
  #121 did not clear RSA's `%rs2233`; #122 supplied the separate correction.
- **#51 / #66:** the closest small-extension pair. One proof uses an explicit
  discard; the other must establish that a named destination has no reads.
  They could share one administrative checklist, but neither existing issue
  wholly subsumes the other's acceptance. #70 already tracks both.
- **#96 / #113 / #118:** scalar table recovery, scalar field demand and mixed
  pointer/integer vector lanes are different proofs. #118 still failed after
  the two earlier corrections; do not close it as another “pointer load” bug.
- **#77 / #78 / #123 / #125:** helper mutation classification, local alias
  classification, helper buffer arguments and registration discovery form
  related stages. Fixing one does not establish correct initialization,
  persistence and binding in the others.
- **#115 / #124 / #127:** latency, allocation failure and a proposed source-size
  optimization are not three proved instances of one root cause. Attribute
  them with controlled measurements before consolidating.
- **#80:** the follow-up replayed both originals and isolated missing backward
  pointer recovery through helper `cvta.to.global`; explicit-pointer and move
  controls emit MSL. Keep this correction and GPU acceptance independent.

## Evidence inventory and limits

Snapshot: `.cumetal-artifacts/duplicate-audit-20260916/` contains `issues.json`,
`prs.json`, `comments.json`, individual Markdown snapshots, `audit.json`, and
`manifest.json`. These are local ignored evidence, not public attachments.
The issue query covers all 68 open author-owned issues; all captured comment
connections report no next page. The PR inventory includes bodies and exact
head commit IDs. That inventory preserves the pre-mutation snapshot. The follow-up records the
subsequent #46 closure, body edits and research-comment receipts separately.

The arithmetic and row coverage were checked against the snapshot: 64 unique
rows, 51 explicit proposed closing owners, eight multi-issue PRs covering 24
unique issue IDs. This audit establishes documentation/scope relationships;
only controlled replay can settle the unresolved shared-root hypotheses.
