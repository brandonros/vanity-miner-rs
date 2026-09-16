# CuMetal #76: implementation and acceptance plan

Status: **implemented locally; acceptance remains open**. The correction is in
`../.worktrees/cuda-metal-issue-76`, branch `upstream/ptx-ssa-type-contract`.
The final candidate passes the focused Debug/Release and PTX functional checks;
Solana, Bitcoin and Ethereum emit MSL under both LLVM versions. This does not
establish downstream GPU correctness. See the
[implementation report](cumetal-76-implementation.md) for exact identities,
measured stages and remaining gates.

Implementation base: CuMetal `9e3e61574b776424a96c686bdbdc04ad1f27fe9f`.
Owner: [cuda-metal #76](https://github.com/Lulzx/cuda-metal/issues/76).

## What must change

A PTX register name can be assigned several different values. An offset held in
`%rd2` can later be replaced by a pointer in `%rd2`. Each assignment needs its own
type. The later pointer must not change the type of an earlier offset carried
through a branch or loop.

The pinned baseline importer violates that contract in three connected places:

1. **Result inference:** `cvt.u64.u32` is initially inferred as a 32-bit result
   because the generic opcode scanner sees the source width. Translation later
   changes the result to 64 bits, after other values and block arguments have
   already copied the earlier type.
2. **CFG copies and rewrites:** per-definition types are keyed by the original
   instruction object's address. Normalization creates new instruction objects.
   Missing metadata falls back to the register name's final type.
3. **Branch/loop joins:** incoming block arguments are seeded from that final
   register-wide type instead of the actual assignments reaching the join.

Relevant pinned source:

- [Result helpers and conversion parser](https://github.com/brandonros/cuda-metal/blob/9e3e61574b776424a96c686bdbdc04ad1f27fe9f/compiler/ir/src/ptx_importer.cpp#L64)
- [Inference and per-destination metadata](https://github.com/brandonros/cuda-metal/blob/9e3e61574b776424a96c686bdbdc04ad1f27fe9f/compiler/ir/src/ptx_importer.cpp#L1666)
- [Allocation fallback](https://github.com/brandonros/cuda-metal/blob/9e3e61574b776424a96c686bdbdc04ad1f27fe9f/compiler/ir/src/ptx_importer.cpp#L1821)
- [Join argument allocation](https://github.com/brandonros/cuda-metal/blob/9e3e61574b776424a96c686bdbdc04ad1f27fe9f/compiler/ir/src/ptx_importer.cpp#L1893)
- [Late conversion correction](https://github.com/brandonros/cuda-metal/blob/9e3e61574b776424a96c686bdbdc04ad1f27fe9f/compiler/ir/src/ptx_importer.cpp#L3084)

## Required invariants

The objective is to eliminate the class of register-wide type contamination.
The following must hold throughout the supported PTX import path:

1. Every assignment and destination lane owns a distinct SSA value. A register
   name identifies mutable PTX storage, not one semantic type for all assignments.
2. Storage width from declarations, operation source interpretation, operation
   result type, and pointer/value evidence are distinct facts. A 64-bit container
   can hold an integer, floating-point bits or a pointer without making every
   assignment have the same semantic type.
3. Every supported result rule is explicit. Input-dependent rules consume actual
   reaching SSA operands. Missing information is never repaired by the final
   type associated with the register name, an arbitrary i32, or source order.
4. Every normalized instruction has complete, correctly ordered result metadata.
   A clone has new result identities. Origin information supports diagnostics
   and applicable evidence; it does not merge the clone with the original.
5. Every join is checked against every executable incoming edge. Definedness,
   compatible types and any permitted edge conversion are separate obligations.
6. The same program has the same type outcome after harmless block layout or
   register-name changes. Identifiers must not override declaration/operation
   semantics; audit name-based storage-width shortcuts where these paths use them.
7. Before imported IR is materialized, original results and block arguments have
   a resolved, consistent contract. Emission cannot silently retag those values.
8. Every failure identifies whether the input is undefined, unsupported,
   conflicting, or an internal inference-completeness error.

## One coordinated correction, with reviewable steps

These steps remain one open #76 correction until all #76 acceptance and regression gates pass.
They may be separate commits for review. Completing one step is not completion
of the issue.

### 1. Establish one instruction-result typing contract

In `compiler/ir/src/ptx_importer.cpp`, use the existing conversion destination
parser before SSA allocation and share result rules with translation. Preserve
source signedness, source width, truncation and rounding as separate semantics.
Do not globally change the generic opcode-width scanner: loads and other
instructions have different operand and result rules.

Audit the supported result families that can disagree with their operand types:
conversions, widened arithmetic, predicate comparisons, tuple results and
integer loads. Reuse the existing ordered `vector<Type>` per destination.
Keep memory access width separate from the width of the receiving register.
Compare inference and emitted result types in tests.

Source inspection also found late result changes for wide arithmetic and a
blanket `.f64` result override worth checking against predicate destinations.
These are regression targets; the audit did not execute new reproducers proving
that each is a current workload failure.

### 2. Preserve each normalized definition's identity and evidence

In `ptx_cfg.cpp`, `ptx_tuple_normalization.cpp` and their existing instruction
storage, give clones distinct identities and retain origin information. Carry
only evidence whose proof remains valid in the new context. Even identical
instruction text may receive different reaching operands on a cloned path;
input-dependent types must be resolved again for those operands. A semantic
rewrite must recompute its result contract, including destination count and lane
ordering; blindly copying old metadata is unsafe.

Infer types for the actual normalized instructions. Calling the old inference
function twice is insufficient: it walks `entry->instructions`, not the
normalized `raw_blocks`. Remove the missing-definition fallback to the final
register type or arbitrary `i32`; incomplete metadata must be diagnosed.

Keep the existing per-destination representation. Coordinate its pointer-lane
evidence interface with #118; this plan does not claim to solve #118's distinct
mixed-vector memory-provenance problem.

### 3. Resolve types over the connected SSA graph

Keep the existing liveness, reaching-definition wiring and undefined-edge
checks. Connect SSA values and incoming/backedge values before freezing their
types. Resolve joins, moves, selects, supported pointer arithmetic and other
input-dependent definitions from their actual SSA operand IDs. A shared opcode
decoder that still reads register-wide input summaries is insufficient.
Use a worklist until stable, including mutually dependent loop arguments, then
fold trivial arguments and materialize IR.

The solver must distinguish unresolved type information from an actual conflict.
An unknown runtime value is not an undefined SSA value. A known generic pointer
address space is not an unknown type. Proven zero is a separate value fact;
integer type alone never permits a pointer edge conversion.
Use each reaching definition's type and the instruction's result contract;
an unrelated later assignment to the same register cannot supply evidence.
Preserve valid existing pointer/null and address-space rules. Reject unresolved
or incompatible joins with the relevant incoming definitions in the diagnostic.
Do not synthesize zero values, widen offsets at the failing pointer operation,
or reinterpret every 64-bit value as a pointer.

The propagation rules must make monotonic progress per SSA value. Specify the
type/provenance facts and their allowed refinements before implementing the
worklist. An unseeded cycle stays unresolved; incompatible concrete seeds produce
a conflict. If a resource budget is exhausted, return an explicit diagnostic.
Increasing the current register-summary inference loop's iteration limit is not
a correction for oscillating register-wide summaries.

### 4. Freeze types before materialization

Translation consumes the resolved result types and must agree with them.
Remove opportunistic destination-type repairs after block argument types have
been embedded. Existing legal source-width extraction and explicit bitcasts
remain in the value builder. Translation may introduce fresh, correctly typed
intermediate values; it may not silently change an existing imported value's
type. Keep verifier checks enabled.

Later documented legalization or helper address-space specialization may
transform types, provided it consistently rewrites affected definitions, uses
and edges and verifies the resulting IR. Freezing importer results does not
prohibit those explicit, checked transformations.

This is a correction to the existing importer and normalization/SSA boundary;
it does not require a replacement compiler IR or a parallel type system.

## Exhaustive audit of the affected boundary

Before coding, enumerate every writer and reader of `register_types`,
`definition_types`, `value_types`, inferred pointer evidence and zero-value
evidence. Classify each use as declaration/container information, an input to a
constraint, a solved result, or a forbidden register-wide fallback. The completion
review must account for each site, including both kernel and helper/inline import
entry paths. Keeping one unreviewed fallback leaves this defect family open.

| Surface | Required treatment |
| --- | --- |
| Conversions | Decode destination and source separately, covering every supported width, category, signedness and rounding modifier. Keep numeric conversion distinct from equal-width bit reinterpretation. |
| Arithmetic | Explicit result rules for ordinary, high-half and wide operations, including carry/multiple-result forms where supported. Type pointer arithmetic only for supported operand combinations. |
| Comparisons and selects | Predicate results remain predicates regardless of operand type; select/copy results use actual reaching input values. Preserve supported pointer/null rules. |
| Loads, stores and parameter slots | Keep memory/ABI byte width separate from destination-container width. Preserve signed/unsigned extension and proven pointer loads without allowing a later register reuse to supply that proof. |
| Tuples and multiple destinations | Per-lane result types, destination ordering and metadata count must agree after splitting, packing and rewriting. Do not collapse a result vector to one guessed scalar type. |
| Declarations, built-ins and implicit bindings | Supply their explicit contracts. Missing implicit metadata must be diagnosed; names or an i32 default cannot replace a known declaration/signature. |
| Helpers, call results and inline PTX | Run the same result rules and propagation, seeded by existing call/return signatures and parameter-slot contracts. Retain generic-pointer specialization and ABI checks. |
| CFG normalization | Inventory clone, split, predicate/selection rewrite, tuple rewrite and tail-call creation sites. Every changed definition has valid metadata; changed operands invalidate dependent facts. |
| Analysis lifecycle | A change to edges, instructions or definitions invalidates affected liveness/reaching/type facts, or uses a verified typed rewrite. Never reuse a stale solution after a mutation. |
| Materialization and later passes | Check the solved result contract on import and verify after relevant transformations. Conversion intermediates get fresh IDs; existing values are not opportunistically retagged. |

This audits the complete supported result-rule surface; it does not promise new
support for currently unsupported PTX instructions or operand combinations.
Unsupported inputs must retain explicit diagnostics.

Keep neighboring defects scoped: #118 owns additional mixed-vector memory
provenance; predicate/definedness and discarded-bit proofs have their own CFG
owners; #123/#125 own helper-global threading/registration. The type solver must
preserve and consume valid evidence at those boundaries. It cannot assume a
missing definition exists or claim to repair a separate proof by assigning a type.

## Required tests

| Gate | Coverage and existing home |
| --- | --- |
| Instruction contract | `tests/unit/ptx_ir_msl_test.cpp`: widening/narrowing, signed/unsigned and integer/float conversions, wide results, predicates, tuple destinations and load-width rules. Assert result and use types directly. |
| Normalized definitions | Same instructions before/after real cloning and semantic rewrites; renamed registers and reordered blocks; scalar → pointer → scalar register reuse. |
| Joins and cycles | `tests/unit/ptx_cfg_test.cpp`: diamonds, empty/copy blocks, loop entry/backedges/exits, converted offsets joining before pointer reuse. Assert every incoming-edge/argument type. |
| Synthesized control flow | Retain local/scalar tail-call, packed-loop and trivial-block-argument regressions; add conversion/reuse combinations. |
| Rejection behavior | Undefined incoming values, incompatible scalar/pointer joins and address spaces, illegal pointer arithmetic, and mismatched widths without a legal conversion must still fail. |
| Numerical GPU checks | Extend `run_ptx_register_reuse.py` and `run_ptx_integer_widths.py`. Preserve the original and joined ReLU cases: all 65 inputs, both backends, exact output, ABI, input and guard checks. Include high-bit values, truncation and signed cases with independently calculated expected results. |

Useful boundary values include `0`, `64`, `0x80000000`, `0xffffffff` and values
above the 32-bit range. Large arithmetic values should be checked as arithmetic;
memory-access probes must remain inside initialized buffers.

For each of the three diagnosed defects, retain a focused case that fails on the
unchanged compiler and passes with the correction. Check typed IR and numerical
output where applicable. A passing existing test that never exercised the defect
does not establish regression coverage. Include block-order and register-renaming
variants so source layout cannot supply accidental type information.

### Coverage across combinations

Testing each ingredient in isolation is insufficient. Build a coverage manifest
for the supported instruction families and the graph boundaries below. Exercise
their interactions explicitly, and explain exclusions as unsupported or
inapplicable instead of treating missing coverage as a pass.

- **Result/source combinations:** supported widths, signed and unsigned sources,
  integer/float categories, bit reinterpretation versus numeric conversion,
  truncation with nonzero upper bits, rounding and extension rules. Include wide
  arithmetic and floating-point comparisons whose results are predicates.
- **Definition forms:** original, copied, cloned with different incoming operands,
  semantically rewritten, multi-result with changed lane ordering, helper-return
  and inline-PTX values. Two instances sharing a source line/text must still own
  different SSA definitions.
- **Control flow:** forward/backward block layout, diamonds and nested joins,
  empty/copy blocks, more than two predecessors, loop preheaders/backedges/exits,
  mutually dependent and nested loops, and synthesized tail-call paths. Include
  zero-, one- and multiple-iteration numerical cases with bounded execution.
- **Register reuse:** scalar → pointer → scalar, conversion → copy → join,
  differing same-typed incoming definitions, and later assignments that must not
  change earlier uses. Rename identifiers while preserving declarations.
- **Pointer/value facts:** valid pointer-plus-offset and pointer-minus-offset,
  pointer/proven-null joins, nonzero or unproved-integer rejection, compatible
  generic/concrete helper spaces, incompatible concrete spaces and per-lane
  results. Preserve existing supported/unsupported operation boundaries.
- **Negative paths:** absent definitions on one incoming edge, unseeded cycles,
  conflicting type seeds, mismatched tuple counts and unsupported operations.
  None may be accepted by defaulting a value or suppressing verification.

Add bounded generated-PTX cases for these combinations, using independent host
expected values for the supported semantics. Keep deterministic seeds and
minimized reproducers in CI. Metamorphic variants (renaming and harmless block
reordering) must preserve numerical outcomes and accepted/rejected status.
Generation supplements the named regressions; it does not replace them.

Add structural completeness checks: every destination has metadata, every use
refers to the correct definition, every edge agrees with its argument contract,
and materialization agrees with solved result types. Exercise these checks with
malformed internal fixtures so they are shown to detect missing metadata.

Run focused Debug and Release checks, then the applicable full repository and
binary-shim-off checks. Compare failures against an unchanged baseline; record
pre-existing failures instead of silently treating them as passing.

## Full-input regression and downstream acceptance

Freeze the new `artifacts/ptx-bundle-llvm7.tar.gz` and
`artifacts/ptx-bundle-llvm21.tar.gz`, built from `7484a5d` plus the recorded timing
instrumentation. Preserve archive/module hashes, selected entry and compiler/
runtime hashes. Compare baseline and candidate on these exact same bytes.
Build the validation CLI from the matching producer source, or document verified
ABI compatibility before execution; retain its binary hash and any dirty inputs.

There are **16 logical modules per LLVM version, 32 PTX inputs total**. Track both
versions separately. The frozen historical RSA modulus artifact has four entries:
its full sweep requires **19 entries per version, 38 total**. Its original translation
script omitted `--entry` and only selected `kernel_rsa_advance`; that result cannot
establish all four stages translated. The current source instead exports only
`kernel_rsa_modulus_vanity` and the updated script explicitly selects and validates
each entry. For that source, rebuilt bundles must contain **16 entries per version,
32 total**, verified against the entry manifest. Retain the frozen artifacts for
historical comparisons and rebuild before testing the new miner.

For each input retain PTX import, Metal emission, Apple compilation, pipeline
creation, GPU completion and CPU verification as distinct stages. Preserve known
passing baseline cells. A failed cell must retain its first diagnostic and known
owner or explicit unassigned status. Similar diagnostics do not prove duplicates.

The new LLVM 7 Solana input provides an additional concrete regression lead:
`mul.hi.u64` at PTX line 34935 consumes `%rd26682`; one reaching assignment is
`cvt.u64.u32` at line 34928 before `$L__BB1_66`, while other definitions are
64-bit shifts. Reduce and inspect its incoming types before claiming it is the
same defect. The passing LLVM 7 Bitcoin/Ethereum MSL cells are regression controls
and candidates for subsequent Apple compilation/GPU validation; they do not
establish that #76 is fixed for the LLVM 21 inputs.

**Close #76 only when** its original joined case, complete typing contract,
normalization/loop/rejection regressions and numerical gates pass, and replay of
the full Solana/Bitcoin/Ethereum modules clears the diagnosed type failures.
Any newly exposed failure within this type contract remains #76 work.

**Close downstream #23/#24/#25 only when** each complete production workload
finishes Metal compilation and GPU execution with selective nonempty prefix/
suffix fixtures, known matches/nonmatches, CPU-verified counts and winners,
independent address reconstruction, unchanged inputs and intact guards across
at least two batches. Passing translation alone does not satisfy this gate.

An independently demonstrated unrelated blocker gets its own existing or new
upstream owner while the downstream workload stays open. No promise is made that
#76 fixes all later stages or all unrelated self-test failures.

## Ownership and integration

Use one compiler implementation owner for the importer, CFG metadata and join
solver; independent reviewers can prepare tests and inspect evidence. Integrate
overlapping #118/CFG edits serially. Use an isolated implementation checkout and
build directory with an explicitly recorded base commit, and one integration owner for
pin updates and GPU validation. Keep the reviewed compiler/runtime pair together.

Publish a PR acceptance table with each gate marked pending/pass/fail, exact
commit/artifact identities and downstream next stage. A passing reducer, included
commit or disappearing first error cannot be summarized as "the kernels work."

## Completion checklist

- [ ] Enumerate and review every affected type/evidence writer and reader, both
  import paths, and every normalization definition-creation site.
- [ ] Implement one authoritative result contract using actual reaching values;
  remove register-summary/default-type authority from SSA definitions and joins.
- [ ] Preserve clone identity, invalidate context-dependent facts and validate
  destination count/order after rewrites.
- [ ] Specify and test convergent propagation, loop cycles, explicit conflicts,
  null proofs, address-space constraints and resource-limit diagnostics.
- [ ] Enforce the import/materialization boundary and checked later transforms;
  retain all strict definedness, ABI and invalid-operation rejection behavior.
- [ ] Make focused regressions fail on the unchanged baseline and pass on the
  candidate; run the named and generated coverage matrix and numerical GPU gates.
- [ ] Replay all 32 frozen PTX inputs with all 38 entries explicitly selected,
  retaining baseline passes, exact hashes and per-stage evidence. Missing or
  blocked checks remain visible; they cannot be counted as passing gates.
- [ ] Also rebuild and freeze the current miner's production/self-test corpus
  for both LLVM versions, verify its complete entry inventory, and compare the
  baseline and candidate compiler on those same new bytes. Use the matching host
  for GPU acceptance; do not mix old four-entry RSA artifacts with the new ABI.
- [ ] Clear every demonstrated remaining failure within the #76 contract. Link
  independent downstream blockers with their evidence and keep the associated
  workload issues open until full GPU acceptance.
- [ ] Review and publish one coordinated correction with its complete acceptance
  table; intermediate commits remain partial implementation until these gates
  establish the claimed scope.

Here, exhaustive means coverage of the supported result-rule families, their
relevant interactions and every boundary that creates or propagates a value's
type. It is not a proof over every possible PTX program. New failures must be
classified against this written contract, not dismissed because a smaller test
already passed.
