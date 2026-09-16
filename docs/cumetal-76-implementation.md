# CuMetal #76 implementation and validation

**The typing correction is published in [draft PR #131](https://github.com/Lulzx/cuda-metal/pull/131). Final issue acceptance remains open.**
The final snapshot passes its selected compiler and numerical GPU regression
suites. Solana, Bitcoin and Ethereum from the fresh miner build emit Metal
source under both LLVM versions. This is not a claim of full workload GPU success.

## Implementation identity

- Worktree: `../../.worktrees/cuda-metal-issue-76`.
- Branch: `upstream/ptx-ssa-type-contract`.
- Base commit: `9e3e61574b776424a96c686bdbdc04ad1f27fe9f` (existing PR #122 stack).
- Published commit: [`c4e5fac44e92ffdd6dae19c1ac62755c814e36b8`](https://github.com/brandonros/cuda-metal/commit/c4e5fac44e92ffdd6dae19c1ac62755c814e36b8); its parent-to-head diff exactly matches the validated candidate 12 source diff.
- Upstream owner: [cuda-metal #76](https://github.com/Lulzx/cuda-metal/issues/76).
- [Implementation contract](cumetal-76-plan.md).
- Evidence root: `../../upstream-issue-breakdown/issue-76-implementation/`.
- Frozen candidate: **12**; binary shim disabled in both Release and Debug.

| Artifact | SHA-256 |
| --- | --- |
| `cumetalc` | `99396dc72b75f133e489bdd6b6415e7aad022069939b175751ab19f93e065a95` |
| `libcumetal.dylib` | `0767c5da3556b7db371ef853a5f52f3e36befc634fa041412640461060c6a2da` |
| Exact source diff | `77762a58d34c6938662b2bd4f1639e148ce191f0bb8e3db8c13567359938e205` |

The contribution is committed and pushed to `brandonros/cuda-metal`, branch
`upstream/ptx-ssa-type-contract`, and proposed upstream as draft PR #131. Issue #76
links the PR and explicitly retains its open acceptance gates. The PR depends
on #122 and its existing stack; its body links the single new commit separately.
The miner's CuMetal pin has not been changed. These are compiler-development
measurements using the explicitly identified compiler/runtime pair; they do not
claim that a pinned miner CLI was rebuilt or bypass its package-identity checks.

The PR contains a self-contained public validation summary. Paths into the local
evidence directory below refer to retained workspace artifacts, not public
attachments.

## What changed

A PTX register name can hold an integer and later a pointer. The importer now
assigns a contract to each actual SSA definition—the individual assignment—and
resolves branch and loop inputs from those definitions.

| Area | Implemented correction |
| --- | --- |
| Result contracts | Conversion source/result types are separate; widened arithmetic, predicates, tuple lanes and load-container widths have explicit rules. Register declarations remain storage contracts. |
| Normalized control flow | Clones own new definitions. A dependency worklist resolves actual operands and loop backedges, with explicit errors for undefined, conflicting or unresolved values. |
| Calls and aggregates | Return slots participate in SSA. Argument-slot evidence requires compatible signatures and stores. Aggregate copies preserve their selected value, private allocation and mutations. |
| Pointer evidence | Generic dereference does not imply device memory. Local-cell evidence must agree with reaching stores on every incoming path; reordered source blocks cannot choose a pointer space. |
| Memory bounds | Finite masked indices and narrowly proved bounded loops establish store ranges. Name summaries require closure across every definition. Partial writes, bypassed bounds, unsafe narrowing and skipped address updates cannot supply false evidence. |
| Emission and verification | Imported result contracts are frozen before emission. New conversions get fresh values. Branch arguments are verified for type, definedness and dominance. |
| Float conversion | Signed destination interpretation and integer rounding modes survive Metal emission. Numerical coverage is finite and in range. |

## Final selected checks

| Check | Candidate 12 result |
| --- | --- |
| Release selected unit tests | **44 passed** |
| Debug selected unit tests, binary shim OFF | **44 passed** |
| Publication check: Debug selected unit tests, binary shim ON | **47 passed** |
| Publication check: Debug shim-ON PTX / typed CUDA groups | **49 + six passed**, one skip in each group |
| PTX functional groups | **49 passed, one skipped** |
| Typed-PTX CUDA projects | **Six passed, one skipped** |
| New bounded pointer-table GPU cases | **Four passed**, 65 inputs each; exact output, ABI and input/output guards |
| Float conversion GPU cases | **20 passed**, including all four integer rounding modes and signed/unsigned destinations |

Commands, exits and full logs are linked from
[candidate-12-test-results.json](../../upstream-issue-breakdown/issue-76-implementation/candidate-12-test-results.json).
[Test identity checks](../../upstream-issue-breakdown/issue-76-implementation/candidate-12/test-identity.json)
confirm the tested build binaries and source diff match the frozen snapshot.
The [coverage inventory](../../upstream-issue-breakdown/issue-76-implementation/coverage.md)
and [type/evidence audit](../../upstream-issue-breakdown/issue-76-implementation/type-site-audit.md)
list the named positive and negative regressions.

The publication check additionally built Debug with the binary shim enabled,
as required by upstream's contribution guide. Its [commands/results](../../upstream-issue-breakdown/issue-76-implementation/publication-validation/results.json)
and [GPU provenance assertions](../../upstream-issue-breakdown/issue-76-implementation/publication-validation/gpu-provenance.json)
confirm the same source diff, with exact launch/pass counts of 7 register-reuse,
9 loaded-pointer, 34 conversion, 6 widening, 3 parameter, 3 store and 6 load cases.
Every counted launch reports Apple GPU execution and generic PTX lowering.
The original candidate 12 binary identities above remain the Release shim-OFF
pair used for the full-corpus and preparation measurements.

`unit_cumetal_cli` is excluded because its readiness check requires unavailable
offline Apple tools. Offline metallib coverage is skipped; wide atomics are also
skipped. Runtime Metal source compilation and numerical GPU execution are available.
Five failures in an earlier broader compiler-suite run reproduced on the
unchanged baseline with the same unavailable-toolchain diagnostics; that
[comparison](../../upstream-issue-breakdown/issue-76-implementation/compiler-functional-baseline-gpu/README.md)
is retained separately and is not counted as a passing final suite.

## Full PTX inputs

Two corpora are kept separate:

| Corpus | Producer and inventory |
| --- | --- |
| Fresh | Miner `afe80210ea28748cc58c3ba75f877bfa4a6b1ecd`, successful [Actions run 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652). **32 modules / 32 explicit entries**: 16 per LLVM version. RSA production has one entry. Artifact ZIP hashes match GitHub's published digests. |
| Historical | Miner `7484a5d` plus recorded timing instrumentation. **32 modules / 38 explicit entries**: historical RSA production has four entries per LLVM version. |

All 32 fresh PTX hashes differ from the historical corpus. Comparisons use the
same PTX bytes for baseline and candidate; results from the two corpora are not
combined or substituted for one another.

### Fresh production targets

| Workload | LLVM7 pinned baseline → candidate 12 | LLVM21 pinned baseline → candidate 12 | Full workload GPU/CPU verification |
| --- | --- | --- | --- |
| Solana | Reject → **MSL emitted** | Reject → **MSL emitted** | Not run |
| Bitcoin | MSL emitted → **MSL emitted** | Reject → **MSL emitted** | Not run |
| Ethereum | MSL emitted → **MSL emitted** | Reject → **MSL emitted** | Not run |

The complete fresh baseline attempted 32 entries: **five emitted MSL, 27 were
rejected, zero timed out**. Candidate 12 attempted the same 32 entries:
**10 emitted MSL, 22 were rejected, zero timed out**. All five baseline MSL
passes were retained. The five additions are LLVM7 Solana; LLVM21 Solana,
Bitcoin and Ethereum; and LLVM21 Ethereum self-tests. Of the 22 remaining
rejections, 15 retain the first diagnostic and seven change it. A changed
diagnostic is not a demonstrated independent defect.

The complete historical comparison covers all 38 entries: baseline **11 MSL /
27 rejected**, candidate 12 **16 MSL / 22 rejected**, with zero timeouts and no
lost baseline MSL passes. The original 32-row baseline is preserved; six
supplemental RSA-entry checks complete its coverage. LLVM21 RSA generate,
ranges and search already passed on the baseline and are retained passes.
Historical RSA-PSS production and self-test also emit MSL on candidate 12,
restoring the known baseline passes that intermediate memory-proof changes broke.

Evidence: [fresh corpus and provenance](../../upstream-issue-breakdown/issue-76-implementation/current-corpus/afe80210-run-35055622652/README.md),
[complete fresh baseline comparison](../../upstream-issue-breakdown/issue-76-implementation/current-corpus/afe80210-run-35055622652/candidate-12-replay/baseline-comparison/comparison.md),
[fresh candidate replay](../../upstream-issue-breakdown/issue-76-implementation/current-corpus/afe80210-run-35055622652/candidate-12-replay/results.json),
[complete historical baseline comparison](../../upstream-issue-breakdown/issue-76-implementation/frozen-replay-candidate-12/full-baseline-comparison/comparison.md),
[historical candidate replay](../../upstream-issue-breakdown/issue-76-implementation/frozen-replay-candidate-12/results.json).

## Remaining fresh translation failures

All 22 rejected inputs retain their exact commands, hashes and diagnostics.
Read-only inspection groups their first failures as follows; it does not prove
that clearing a first failure would clear the whole module.

| First-failure family | Inputs | What remains |
| --- | --- | --- |
| Undefined incoming value before type solving | 9 | Existing CFG/definedness rejection; unchanged first diagnostic on the same baseline input. Preserve the rejection until the missing path proof is established. |
| Unsupported operation or ABI transfer | 6 | Four `clz.b64` inputs, one vector parameter transfer and one scalar-minus-pointer expression. These also reject on the unchanged baseline. |
| Bitwise address construction | 5 | LLVM7 pointer addresses formed with `or.b64` become integer results and later meet pointer values at joins. Establish a sound bitwise-address proof/legalization; do not retag arbitrary integers as pointers. |
| Local-cell memory evidence | 2 | LLVM7/21 Shallenge self-tests need branch/iterator range and alias proofs to establish that writes cannot overlap a stored pointer. The new guard rejects without that proof. |

The last two rows remain open pointer-provenance/memory-evidence acceptance
work. They are not evidence of the original conversion-result width bug, and
they are not declared unrelated or solved. Exact source chains and next actions
are in the [two LLVM7 cases](../../upstream-issue-breakdown/issue-76-implementation/candidate-12-residual-llvm7-two.md)
and [other 20 cases](../../upstream-issue-breakdown/issue-76-implementation/candidate-12-residual-other-20.md).

## Apple preparation and remaining acceptance

Candidate 12's paired runtime was measured serially on the three fresh LLVM21
MSL outputs, after all translation jobs finished. Workload specializations were
disabled. The harness forces library/pipeline creation through
`cuFuncGetAttribute(MAX_THREADS_PER_BLOCK)`; successful `cuModuleLoad` alone
does not establish compilation.

| Production entry | Preparation result | Elapsed wall time | Kernel launched |
| --- | --- | --- | --- |
| Solana | **Prepared**, maximum threads per block 1024 | 14.485 s | No |
| Bitcoin | **Timed out** inside `cuFuncGetAttribute` | 90.058 s | No |
| Ethereum | **Timed out** inside `cuFuncGetAttribute` | 90.055 s | No |

The two timeouts establish that preparation did not finish within the limit.
These API-level logs do not isolate Metal library compilation from pipeline
creation, prove an out-of-memory failure, or establish whether a longer run
would succeed. The harness terminated its timed-out process groups. No
production kernel was launched by this measurement; Solana preparation is not
production GPU/CPU verification.

[Preparation results, exact hashes and stage logs](../../upstream-issue-breakdown/issue-76-implementation/apple-preparation-candidate-12-fresh/results.json)
identify the final compiler/runtime/MSL artifacts. Earlier snapshots' preparation
results remain separate and are not attributed to candidate 12.

Full Solana/Bitcoin/Ethereum acceptance still requires the matching miner host/ABI,
Apple compilation, pipeline creation, GPU completion and independent CPU checks
with selective patterns and repeated batches. Passing reducers or writing Metal
source does not close downstream #23/#24/#25.

The legacy backend still cannot emit the joined ReLU CFG fixtures; those cells
are explicitly `NOT_TESTED`, not passing. Exceptional floating-point inputs,
saturation, out-of-range conversions and every wider-container combination are
not established by the finite numerical cases. Remaining full-module diagnostics
must keep their actual stage and owner; a changed first error is not proof that
another issue is unrelated or solved.
