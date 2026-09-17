# CuMetal issue ownership and fix status

**Current pin: `fb3644a251befdc4c1a9d82a8742ae04ae2498c4`.**
[Draft PR #142](https://github.com/Lulzx/cuda-metal/pull/142) implements the later
#76 proof-scaling follow-up, on #139/#138/#135 and the preceding contribution
stack. The commit is pushed; GitGuardian passes at its exact head. The matched
Nix compiler/runtime and normal all-mode miner were rebuilt and replayed.

## Latest measured changes: #76 proof gaps cleared

The [consumer report](cumetal-76-scaling-validation.md) records **53 CPU checks
passed** and two unchanged LLVM7 self-test module attempts:

| Workload | Normal pinned CLI | Next owner |
| --- | --- | --- |
| Bitcoin self-tests / #30 | Prior prefix-proof exhaustion cleared; rejects helper call after **94.244s** | [#136: helper-write footprints](https://github.com/Lulzx/cuda-metal/issues/136) |
| RSA-PSS self-tests / #35 | Prior address-demand budget cleared; rejects `clz.b64` after **80.859s** | [#141: direct CLZ](https://github.com/Lulzx/cuda-metal/issues/141) |

Both finish before the 600-second deadline, before Apple compilation/GPU
execution. All 53 selected names are reported once as compilation-blocked.
Neither downstream issue closes. These are two current-pin attempts, not a
fresh 32-combination sweep. Other rows retain their named measured revisions.

**Why #76 remains open:** its original conversion/SSA repair (#131), alignment/
local-memory follow-up (#135), and later proof-scaling correction (#142) are
implemented and pushed. The outstanding items are native legacy numerical
acceptance requiring unavailable offline Apple tools, plus review/integration
of the pending stack. Bitcoin's next helper proof is #136; RSA-PSS's next
instruction gap is #141. Neither is an unfinished copy of the original type fix.

## Workload count and implementation order

Live GitHub audit: **5 closed workload issues (#23–25,#38,#39), 11 open**:
three production modes (P-256 signature, RSA modulus, RSA-PSS), and all eight
self-test groups. Across LLVM7/LLVM21, the 16 workloads have 32 combinations:
**10 have named-revision acceptance, 22 remain without complete acceptance**.
This is not a claim that 22 combinations were just run and failed. Broad #16/#19
bring the repository-wide open issue count to 13.

The requested implementation order after #76 is **#118 → #140 → #136 → #141 →
#133**: per-lane vector pointer recovery, helper pointee provenance, helper-write
footprints, direct CLZ, then Ethereum pipeline preparation. #118 now has an
isolated working branch; no fix/GPU acceptance is claimed for it yet. Each
workload retains later-stage acceptance independently of its first compiler fix.

The scoped #76 regressions pass 57 Release/60 Debug selected unit tests and 57
PTX functional tests in each configuration (documented skips/exclusions).
Both configurations pass 16 numerical GPU fixtures  × 65 lanes and 29 refusal
controls. See the report for the corrected negative-fixture construction,
exact identities and the distinction from full-module GPU acceptance.

## P-256 public-key production accepted at `0f98856`; #39 closed

The [normal-miner report](cumetal-130-validation.md) records **11 passing
invocations, 22 GPU batches and 704 CPU-reference-checked candidate positions**,
with 18 selected winners independently reconstructed on the host. Original
LLVM7 input and both latest versions pass; profiles cover all four targets,
selective prefix/suffix patterns, all-match and no-match. Candidates use OS
entropy, not reproducible `--seed` fixtures. Upstream #130 is implemented and
production-verified in an unmerged draft; its workload issue is closed.

## #130 self-test replay at `0f98856`: issues stay open

The [pinned report](cumetal-130-validation.md) records **147 CPU passes** and four
GPU attempts: **78 passing checks, one numerical failure, 68 blocked checks**.
Solana LLVM21 executes but `candidate_match` returns 0 instead of 1. Ethereum
LLVM7 times out in pipeline preparation. Bitcoin LLVM7 reaches a #76 memory-proof
budget; Shallenge LLVM21 reaches #118 mixed pointer/length reload errors in Metal.
None of #29/#30/#31/#37 closes. Solana’s confirmed numerical owner is
[#140](https://github.com/Lulzx/cuda-metal/issues/140): private record-field
pointers are reloaded through helpers as device pointers. A 65-lane GPU reduction
returns 0 instead of 97; four controls pass. The same reduced wrong code is
emitted at `4a207e2`, so it predates both new follow-ups.

#130 is implemented in a published draft with focused GPU regressions passing;
it is no longer “no PR.” The exact later failures remain separately tracked.
#134 narrowing is published in PR #139; its full Bitcoin input advances to
mixed-vector pointer typing (#118), with an additional empty-slice sentinel
proof boundary. The normal consumer confirms 144 pointer/i64 mismatches after
94.578 seconds at `f7ceeef`, before any GPU launch. Other results
below retain their named measured revisions; no new 16-workload score is claimed.

## Earlier #76 follow-up at `13efc29`

At `13efc29`, focused development builds pass **55 PTX functional tests each** in
Release/shim OFF and Debug/shim ON, including numerical address/memory and
conversion checks. GitGuardian passes on the exact published head. Native legacy
GPU acceptance still needs the offline Apple toolchain. Later replays exposed Bitcoin initialized-prefix and RSA-PSS address-demand
budgets, subsequently corrected in #142 at `fb3644a`. Native legacy acceptance
and upstream integration remain open. [Acceptance details](https://github.com/brandonros/cuda-metal/blob/13efc293c6cd06da5616a87b2dd8db81b1b0e0df/docs/issue-76-acceptance-tests.md).

Seven unchanged full-module translation replays clear their preceding #76 gates
but still fail before execution. The three Solana/Shallenge rows are also confirmed through the normal pinned
miner; the other four are development-compiler translation measurements. None
is a GPU pass:

| Module/version | Next measured blocker | Owner |
| --- | --- | --- |
| Solana self-tests / LLVM7 | Helper-call writes may affect a saved pointer cell | [#136](https://github.com/Lulzx/cuda-metal/issues/136), new confirmed precision gap |
| Shallenge self-tests / LLVM7 | Heterogeneous vector pointer reload | [#118](https://github.com/Lulzx/cuda-metal/issues/118) |
| Shallenge self-tests / LLVM21 | Zero-marker guard loses payload definedness | [#130](https://github.com/Lulzx/cuda-metal/issues/130), reduced and confirmed |
| P-256 signature production / LLVM7 | Vector parameter transfer | #41 scope lead; not yet a confirmed duplicate |
| P-256 public-key self-tests / LLVM7 | Undefined `%rd18761` | #130 scope lead |
| P-256 signature self-tests / LLVM7 | Undefined `%rd16384` | Needs reduction before assigning ownership |
| RSA-PSS self-tests / LLVM7 | Undefined `%rd33628` | Needs reduction before assigning ownership |

[#137](https://github.com/Lulzx/cuda-metal/issues/137) separately owns the
preexisting escaped-cell validation omission found by negative controls. It is
not a demonstrated GPU numerical failure. No downstream workload issue closes
from these seven translation results. The newer #130 implementation and replay
are recorded above; #134 is implemented and its later failure is recorded in
the historical `f7ceeef` report. The separate Ethereum pipeline-preparation owner remains
[#133](https://github.com/Lulzx/cuda-metal/issues/133).

## Earlier pin: Shallenge production accepted at `4a207e2`

**Shallenge production now passes both LLVM7 and LLVM21. Downstream #38 is closed.**
Upstream #129 is **implemented and CPU/GPU verified, with its PR still unmerged**.
It remains open for integration, not because the measured workload still fails.
GitGuardian passed on the exact PR head. These results were published and read
back on PR #132, upstream #129 and downstream #38 on 2026-09-16.

## Historical four-group replay at `4a207e2`

The [four-group replay](cumetal-four-self-test-validation.md) freshly attempts
Solana, Bitcoin, Ethereum and Shallenge self-tests under **both LLVM7 and LLVM21
at `4a207e2`**. **None of #29/#30/#31/#37 can close.** All **147 CPU checks pass**;
the eight CuMetal attempts produce seven translation failures and one
**600-second LLVM21 Ethereum pipeline-preparation timeout**. All 294 GPU check
selections are blocked before execution; no numerical failures or skips occurred.
Ethereum stack samples at about 120 and 578 seconds show Apple compute-pipeline
creation after successful source-library creation and function lookup.

**All eight measured combinations now have published upstream owners.** #76 owns
LLVM7 Solana and both Shallenge proof gaps; #130 owns LLVM21 Solana and LLVM7
Bitcoin/Ethereum guards. New [#134](https://github.com/Lulzx/cuda-metal/issues/134)
owns LLVM21 Bitcoin narrowing; new [#133](https://github.com/Lulzx/cuda-metal/issues/133)
owns LLVM21 Ethereum pipeline preparation. The upstream and four downstream
issue bodies were updated and read back. See [the published ownership and #76
checklist](cumetal-four-self-test-validation.md#published-upstream-ownership).

## Shallenge production verified on both versions

The [Shallenge report](cumetal-shallenge-validation.md) records **66 passing
invocations, 132 GPU batches and 4,224 CPU-reference-checked candidate positions**.
Both versions pass all 30 valid username lengths, seed controls and a no-match
case. The CPU checks aggregate counts and selected winner payloads; independent
SHA-256 checks verify all 96 published winners. Positions are not unique-input
counts, and the kernel does not export every candidate payload.

LLVM7 and LLVM21 first invocations took **3.443 s and 1.259 s** respectively;
no run reached its 600-second deadline. Both original PTX inputs are unchanged.
The compiler proves that a shared allocation address cancels to a scalar count;
it no longer rejects LLVM7's intermediate `1 - cursor` before that proof.
The separate Shallenge self-tests (#37) remain open: LLVM7 rejects translation
at `13efc29`; LLVM21 reaches Metal pointer-cast errors at `0f98856`.

## Earlier verified production results

**Solana, Bitcoin and Ethereum passed both versions at `0242f22`; #23–25 remain
closed for that acceptance.** The [three-mode report](cumetal-three-mode-validation.md)
records 24 invocations, 48 GPU batches and 1,536 verified candidate positions.
Their four profiles per mode/version cover two selective seeds, no-match and
all-match. They were not rerun at `4a207e2`.

Their long first-run Bitcoin/Ethereum waits were sampled in Metal pipeline
creation; all completed within 600 seconds. The GPU command-buffer intervals
were 5.3–26.2 ms per batch. See the [#76 implementation report](cumetal-76-implementation.md)
for the qualifier correction and its independent remaining proof limits.

## Stage matrix: measured revisions are explicit

Inputs: miner **`afe80210`**, [Actions run 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
The #130 measurements use the normal matched consumer at `0f98856`;
LLVM21 Bitcoin uses `f7ceeef`; LLVM7 Bitcoin/RSA-PSS self-tests use current `fb3644a`.
Original LLVM7 P-256 input from `7484a5d` has its separate acceptance above.
Other rows retain their stated revisions. No current-pin 16-workload score
combines those historical results with the new rows.

The earlier full translation sweep measured LLVM7 **3/16** and LLVM21 **7/16**
at `c4e5fac`. Those totals are historical; no current-pin 16-workload score is claimed.

| Workload / downstream tracker | Translation revision | LLVM7 PTX → Metal | LLVM21 PTX → Metal | Apple preparation at named revision | GPU + CPU at named revision |
| --- | --- | --- | --- | --- | --- |
| Solana production ([#23](https://github.com/brandonros/vanity-miner-rs/issues/23)) | `0242f22` | **Pass at `0242f22`** | **Pass at `0242f22`** | **Both versions passed** | **Both versions passed all four profiles** |
| Bitcoin production ([#24](https://github.com/brandonros/vanity-miner-rs/issues/24)) | `0242f22` | **Pass at `0242f22`** | **Pass at `0242f22`** | **Both versions passed** | **Both versions passed all four profiles** |
| Ethereum production ([#25](https://github.com/brandonros/vanity-miner-rs/issues/25)) | `0242f22` | **Pass at `0242f22`** | **Pass at `0242f22`** | **Both versions passed** | **Both versions passed all four profiles** |
| Shallenge production ([#38](https://github.com/brandonros/vanity-miner-rs/issues/38), closed) | `4a207e2` | **Pass at `4a207e2`** | **Pass at `4a207e2`** | **Both versions passed** | **Both versions passed all 33 profiles**; 132 GPU batches / 4,224 positions total |
| P-256 public-key production ([#39](https://github.com/brandonros/vanity-miner-rs/issues/39), closed) | `0f98856` | **Pass; original and latest inputs** | **Pass** | Both versions pass | **Both versions pass all five profiles; original LLVM7 also passes** |
| P-256 signature production ([#26](https://github.com/brandonros/vanity-miner-rs/issues/26)) | LLVM7 development `13efc29`; LLVM21 historical `c4e5fac` | Rejected: vector parameter transfer; #41 is a scope lead | Rejected: undefined edge | Not rerun | Not rerun |
| RSA modulus production ([#27](https://github.com/brandonros/vanity-miner-rs/issues/27)) | `c4e5fac` (historical) | Rejected: undefined edge | Rejected: `clz.b64` unsupported ([#141](https://github.com/Lulzx/cuda-metal/issues/141)) | Not rerun | Not rerun |
| RSA-PSS production ([#28](https://github.com/brandonros/vanity-miner-rs/issues/28)) | `c4e5fac` (historical) | Rejected: vector parameter transfer | Rejected: `clz.b64` unsupported ([#141](https://github.com/Lulzx/cuda-metal/issues/141)) | Not rerun | Not rerun |
| Solana self-tests ([#29](https://github.com/brandonros/vanity-miner-rs/issues/29)) | LLVM7 `13efc29`; LLVM21 `0f98856` | Rejected: helper-write proof (#136) | **MSL emitted** | LLVM21 passed in 405.087 s total | **LLVM21: 78 pass, `candidate_match` fails (0 vs 1), owner #140**; CPU 79/79 pass |
| Bitcoin self-tests ([#30](https://github.com/brandonros/vanity-miner-rs/issues/30)) | LLVM7 `fb3644a`; LLVM21 `f7ceeef` | #76 cleared; rejected helper-write proof (#136), 94.244s | #134 clears; 144 later pointer/i64 mismatches (#118 plus empty-slice proof) | Not reached | CPU 39/39 pass; GPU blocked at named revisions |
| Ethereum self-tests ([#31](https://github.com/brandonros/vanity-miner-rs/issues/31)) | LLVM7 `0f98856`; LLVM21 `4a207e2` | **MSL emitted** | **MSL emitted** | Both named revisions pass library creation; pipeline preparation times out at 600 s | CPU 8/8 pass; no completed GPU checks |
| Shallenge self-tests ([#37](https://github.com/brandonros/vanity-miner-rs/issues/37)) | LLVM7 `13efc29`; LLVM21 `0f98856` | Rejected: mixed-vector pointer (#118) | **MSL emitted** | LLVM21 rejects mixed pointer/length reload casts (#118) | CPU 21/21 pass; GPU blocked at named revisions |
| P-256 public-key self-tests ([#32](https://github.com/brandonros/vanity-miner-rs/issues/32)) | LLVM7 development `0f98856`; LLVM21 historical `c4e5fac` | Rejected: `%r29315` undefined at `$L__BB19_1`; later case needs reduction | **MSL emitted** | Not rerun | Not rerun |
| P-256 signature self-tests ([#33](https://github.com/brandonros/vanity-miner-rs/issues/33)) | LLVM7 development `0f98856`; LLVM21 historical `c4e5fac` | Rejected: `%rs2098` undefined at `$L__BB35_4`; later case needs reduction | Rejected: undefined edge | Not rerun | Not rerun |
| RSA modulus self-tests ([#34](https://github.com/brandonros/vanity-miner-rs/issues/34)) | `c4e5fac` (historical) | Rejected: undefined edge | Rejected: `clz.b64` unsupported ([#141](https://github.com/Lulzx/cuda-metal/issues/141)) | Not rerun | Not rerun |
| RSA-PSS self-tests ([#35](https://github.com/brandonros/vanity-miner-rs/issues/35)) | LLVM7 `fb3644a`; LLVM21 historical `c4e5fac` | #76 cleared; rejected direct `clz.b64` (#141), 80.859s | Rejected: `clz.b64` unsupported ([#141](https://github.com/Lulzx/cuda-metal/issues/141)) | Not rerun | Not rerun |

The newer development-compiler cells above come from the
[#130 unchanged-input replay](https://github.com/brandonros/cuda-metal/blob/0f98856b9bcab06f0c41d239a86684fdf0d6371d/docs/ptx-scalar-zero-guards-validation.md#unchanged-full-input-translation-replay).
They do not establish Apple compilation or GPU results. The #76 follow-up now
has normal pinned-consumer evidence at `fb3644a` for Bitcoin/RSA-PSS LLVM7.

## Next work and closure

1. **Five production workloads have measured acceptance:** #23–25 at `0242f22`,
   #38 at `4a207e2`, and #39 at `0f98856`. They remain closed on that evidence.
   Their upstream PRs still await integration; these are not fresh current-pin
   regression results.
2. **Four self-test groups remain open after advancing their original blockers.**
   Solana LLVM21 has a confirmed numerical defect [#140](https://github.com/Lulzx/cuda-metal/issues/140);
   LLVM7 needs helper-write proofs (#136). Bitcoin LLVM7 now also needs #136;
   LLVM21 needs per-lane pointer recovery (#118) and sound empty-slice handling.
   Shallenge needs #118. Ethereum needs pipeline-preparation investigation (#133).
   #130/#134 are implemented in PRs #138/#139; do not assign their cleared errors
   as the current first failures.
3. **Other combinations retain older measured revisions.** Only Bitcoin and
   RSA-PSS LLVM7 self-tests were replayed at `fb3644a`; nine other open workloads
   and the LLVM21 counterparts need current-pin testing before assigning fresh
   failures. Their recorded historical acceptance and first failures remain separate.
4. **#76 proof-scaling gates are implemented in #142.** Native legacy numerical
   validation still needs offline Apple tools; review/integration remains. #136
   and #137 separately own helper-write footprints and validation discovery.

The historical `c4e5fac` LLVM21 RSA #27/#28/#34/#35 attempts stopped at unsupported
`clz.b64`; dedicated [#141](https://github.com/Lulzx/cuda-metal/issues/141) now
owns that opcode gap. Its complete reproducer, compiler identities, u32 result
contract and acceptance criteria are published, and ownership updates were read
back on upstream #76 and downstream #27/#28/#34/#35. No implementation PR or
new GPU result is claimed. Historical resource cases remain unresolved. For startup optimization,
[#127](https://github.com/Lulzx/cuda-metal/issues/127) remains a measured-experiment
candidate; source size alone does not establish Apple's dominant compile cost.

**Five workload issues are closed: #23–25, #38 and #39. Eleven remain open:**
three production modes and all eight self-test groups. Broad #16/#19 are separate.
Only Bitcoin LLVM21 was replayed through the normal miner at the current `f7ceeef`
pin. The named-revision stage table deliberately retains earlier measurements;
it does not claim a fresh 16-workload score.

### Unpublished-draft audit

CLZ was the only ready unpublished standalone issue draft; it is now #141.
The other completed drafts already have owners, including #76/#77/#78,
#133/#134, #136/#137 and #140. Bitcoin's additional empty-slice boundary is
already public on [#118](https://github.com/Lulzx/cuda-metal/issues/118#issuecomment-5706131679)
and still needs an isolated reproduction before a separate owner can be assigned.

The additional kernel call-argument demand finding is now preserved as
[research on #137](https://github.com/Lulzx/cuda-metal/issues/137#issuecomment-5707038257).
It passes a loaded value to a reading helper, distinct from #137's original
escaped-cell/writing-helper case. The comment explicitly identifies the extra
acceptance scope if grouped there; it is not a demonstrated blocker among the
22 outstanding workload/version combinations. No further ready issue draft
was found in this audit.

The following audit and full-GPU measurements are historical unless an entry
names an exact newer commit. PR #131 contains both `c4e5fac` and `0242f22`; its
number alone does not identify a measured build. PR #132 adds `4a207e2`.
Use the stage table and linked validation reports for each measured revision.

The [64-issue consolidation audit](cumetal-issue-duplicate-audit.md) and
[executed follow-up](cumetal-consolidation-followup.md) now have published GitHub
results: upstream #46 is **closed as superseded** after transferring acceptance
to #116/#117. #83's historical depth subcase is assigned to #120/#121; its
empty-label work remains open. This leaves 63 of the audited trackers open;
51 have open PRs explicitly claiming fixes, not 51 verified or merged repairs.

Fresh CPU translation controls isolate #80's missing generic-to-global helper
pointer recovery; no implementation or GPU pass is claimed. Comparing the
retained #115/#124 sources confirms shared constant-PRMT expansion and diagnostic
source reductions of 13.21%/31.87%, making #127 a shared optimization target.
Their Metal timeout/allocation causes remain unresolved and separately tracked.
The follow-up links the published evidence and records exact inputs and limits.

RSA source update: production now uses one resumable `kernel_rsa_modulus_vanity`
entry. The RSA GPU measurements below concern the historical four-stage PTX,
not this refactor; the updated RSA self-test also needs fresh GPU validation.
The latest `afe80210` LLVM21 production PTX rejects unsupported `clz.b64` under
both the baseline and `c4e5fac`; the earlier `%rd310` / `$L__BB0_6` report describes
an older input. No new RSA GPU execution is established. See
[the mining design and checks](gpu-search-pipeline.md).

## Historical split-bundle translation check (`7484a5d`)

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
there were **16 open workload trackers at that translation-only measurement**,
one for each logical module. The subsequent production validation closed #23–25.
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
| [#38 — LLVM 7 Shallenge production](https://github.com/brandonros/vanity-miner-rs/issues/38) | `shallenge.ptx`: exit 1 after 0.350 s; line 306, `pointer subtraction requires a pointer minus a 64-bit integer byte offset`. SHA-256 `bfbd79fec26057bfc77293777f3acbcb40cb84c61041274e2c66fe6770639e51`. | LLVM 21 emits MSL in 0.409 s on this bundle. Production is distinct from self-test issue #37. [Upstream #129](https://github.com/Lulzx/cuda-metal/issues/129) now tracks proven same-base address cancellation; the failing integer-minus-pointer form is explicitly outside #45/#56 and is not #76 conversion inference. |
| [#39 — LLVM7 P-256 public-key production](https://github.com/brandonros/vanity-miner-rs/issues/39) | Historical `7484a5d` input: `%rd18761` undefined at `$L__BB5_1`, 9.412 s; SHA-256 `dbd5ada9e96613eb7f2da98e97b95310a5a6368744c5f8d1d84b9c184184e0da`. | At that measurement LLVM21 emitted MSL. Owner [#130](https://github.com/Lulzx/cuda-metal/issues/130) subsequently implemented the guard correction; **#39 is now closed** after normal-consumer acceptance at `0f98856`, recorded above. |


Both published issues retain producer `7484a5d` plus the recorded timing
instrumentation, CuMetal `9e3e615`, the exact compiler and input hashes, and the
full diagnostic. First require the retained LLVM 7 input to translate, then
complete Apple compilation and bounded CPU-verified production batches with
input/buffer guards intact; retain an independent LLVM 21 regression row. At that historical measurement both compiler corrections were unimplemented;
research and complete small reproducers were published in **#129** and **#130**.
The later implementation and workload acceptance for both appear at the top of
this report. These assignments cover the demonstrated first gaps, not
every possible later failure or full-workload acceptance.

The issue bodies include the producer revision, build/translation commands,
exact diagnostics and input hashes. Original `.cumetal-artifacts/` snapshots
remain local evidence, not public attachments; regenerated inputs must have
their hashes checked before being attributed to this run. Do not substitute
the older Actions-produced inputs for these locally generated bundles.

### Published LLVM 7 ownership research

[Research details and probe timings](cumetal-llvm7-ownership-research.md) distinguish
fresh compile-only experiments from the original split-bundle results. Findings
were attached to [downstream #38](https://github.com/brandonros/vanity-miner-rs/issues/38#issuecomment-5691622619)
and [#39](https://github.com/brandonros/vanity-miner-rs/issues/39#issuecomment-5691622840).

- **#38 → upstream #129:** LLVM 7 forms `base + (1 - cursor - length) + 30`.
  The prologue establishes `cursor = base + ((31 - length) & 3)`, so the base
  cancels and the result is a scalar loop count. A compact input reproduces the
  rejected intermediate; a diagnostic full-input scalar rewrite emits MSL. The
  required compiler work is proven address cancellation, not blanket acceptance
  of integer-minus-pointer or suppression of type verification.
- **#39 → upstream #130:** a scalar zero marker guards an optional payload.
  The CFG proof tracks predicate literals but lacks the needed scalar move/zero
  facts. The compact failure has no empty blocks or deep paths, persists with
  unsigned equality, and clears with the original predicate. Full-input guard
  controls clear the first payload demands but expose another undefined register.
  Do not initialize absent payloads or equate all later failures with this scope.

All four valid small forms tested with NVIDIA CUDA 12.9 ptxas were accepted.
Observable-undefined controls remain rejected by CuMetal. No compiler code or
pin was changed, no GPU execution was attempted, and no workload issue was
closed. Original full-input hashes/results remain unchanged. The upstream issue
bodies include complete public small reproducers; local full-input controls and
logs remain in `.cumetal-artifacts/research-38-39/`.

## Recorded full GPU run

Updated **2026-09-16** after the complete fresh run, **01:46:07–02:14:38 UTC**.
The [16-row result](cumetal-status.md) is **2 true / 14 false**: Shallenge and
P-256 public-key production pass; all eight self-test groups are blocked before
execution. That report had **14 workload issues (#23–35 plus #37)**, including
[#37 — Shallenge self-tests](https://github.com/brandonros/vanity-miner-rs/issues/37).
The subsequent LLVM 7 production trackers **#38/#39** bring the current total to
**16**; they do not change the earlier full-GPU measurements.
This tracking update adds no new measurements.

## Source and pin for the historical full-GPU run

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

**Unless a row names a newer commit, GPU evidence below belongs to historical
`9e3e615`. Split-bundle results and the `c4e5fac` translation sweep are separate
measurements. The inclusion column refers only to historical `9e3e615`.**
Existing upstream owners retain their
original reproducer scope. Newly observed diagnostics are assigned only where
supported; research leads are labeled explicitly. Blocked checks have no GPU
numerical result; Solana LLVM21 now has one actual numerical failure, stated below.

| Vanity-miner issue | Upstream ownership | Published implementation | Included in `9e3e615`? | Recorded result and latest next work |
| --- | --- | --- | --- | --- |
| [#23 — Solana production](https://github.com/brandonros/vanity-miner-rs/issues/23) | [#76 / #131](https://github.com/Lulzx/cuda-metal/issues/76); [#113](https://github.com/Lulzx/cuda-metal/issues/113) pointer-load acceptance | **Implemented in `0242f22`; both LLVM versions CPU/GPU verified.** [Measured report](cumetal-three-mode-validation.md). | Earlier PRs only; #131 was not in historical `9e3e615`. | **`0242f22`:** eight invocations / 16 GPU batches across both versions pass CPU counts/payload, guards and independent-address checks. **Closed as completed on 2026-09-16** after publishing and verifying this evidence. |
| [#24 — Bitcoin production](https://github.com/brandonros/vanity-miner-rs/issues/24) | [#76 / #131](https://github.com/Lulzx/cuda-metal/issues/76); [#113](https://github.com/Lulzx/cuda-metal/issues/113) pointer-load acceptance | **Implemented in `0242f22`; both LLVM versions CPU/GPU verified.** [Measured report](cumetal-three-mode-validation.md). | Earlier PRs only; #131 was not in historical `9e3e615`. | **`0242f22`:** eight invocations / 16 GPU batches across both versions pass CPU counts/payload, guards and independent-address checks. **Closed as completed on 2026-09-16** after publishing and verifying this evidence. |
| [#25 — Ethereum production](https://github.com/brandonros/vanity-miner-rs/issues/25) | [#76 / #131](https://github.com/Lulzx/cuda-metal/issues/76); [#113](https://github.com/Lulzx/cuda-metal/issues/113) pointer-load acceptance | **Implemented in `0242f22`; both LLVM versions CPU/GPU verified.** [Measured report](cumetal-three-mode-validation.md). | Earlier PRs only; #131 was not in historical `9e3e615`. | **`0242f22`:** eight invocations / 16 GPU batches across both versions pass CPU counts/payload, guards and independent-address checks. **Closed as completed on 2026-09-16** after publishing and verifying this evidence. |
| [#26 — P-256 signature production](https://github.com/brandonros/vanity-miner-rs/issues/26) | [#83 — empty labels/predicate facts](https://github.com/Lulzx/cuda-metal/issues/83) | #85 covers original predicate cases; **no empty-label follow-up PR**. | #85 and #121 included; #121 addresses a different depth case. | **Both sources fail PTX translation:** `%rd17653` undefined at `$L__BB0_11`. Preserve facts across empty blocks and validate message/ephemeral search. |
| [#27 — RSA modulus production](https://github.com/brandonros/vanity-miner-rs/issues/27) | [#119](https://github.com/Lulzx/cuda-metal/issues/119) owns the **historical GPU-completion wait**. Fresh pipeline-preparation wait: **owner unconfirmed**. | **No completion/performance fix PR.** | No corresponding fix. | **Times out:** fresh samples at about 180 s and near the deadline are inside **Metal pipeline creation**, before verified execution. This is not the historical device-synchronization sample. Attribute the preparation cost and confirm an owner; then complete verified four-stage cycles. |
| [#28 — RSA-PSS production](https://github.com/brandonros/vanity-miner-rs/issues/28) | Earlier #96/#109/#111 → [#115 — Metal compilation time](https://github.com/Lulzx/cuda-metal/issues/115) | #108/#110/#112 clear translation; [#128](https://github.com/Lulzx/cuda-metal/pull/128) adds **diagnostics only**. **No performance fix PR.** | Translation fixes yes; #128 **no**. | **Salt and message searches time out** while sampled in Metal source-library compilation. Both emit source; neither verifies candidates. Diagnose and reduce compilation cost, then validate both sources. |
| [#29 — Solana self-tests](https://github.com/brandonros/vanity-miner-rs/issues/29) | LLVM7: [#136](https://github.com/Lulzx/cuda-metal/issues/136). LLVM21: [#140](https://github.com/Lulzx/cuda-metal/issues/140). | #76/#135 and #130/#138 clear preceding address/guard failures. No #136/#140 fix claimed. | New follow-ups absent. | **LLVM7 `13efc29`:** helper-write proof rejection. **LLVM21 `0f98856`: 78 pass, one numerical failure (`candidate_match`: 0 vs 1).** CPU 79/79 pass. Private record fields lose their pointee address space through helper loads; published 65-lane reduction and four controls establish #140 ownership. [Report](cumetal-130-validation.md). |
| [#30 — Bitcoin self-tests](https://github.com/brandonros/vanity-miner-rs/issues/30) | LLVM7: [#136](https://github.com/Lulzx/cuda-metal/issues/136) helper-write footprints after #142 clears the proof budget. LLVM21: [#118](https://github.com/Lulzx/cuda-metal/issues/118) per-lane reloads; empty-slice proof is an additional boundary. | **#130/#138 and #134/#139 implemented; original guard/narrowing errors cleared.** | New follow-ups absent. | CPU 39/39 pass. **LLVM7 `fb3644a`:** #76 clears; helper call24791 rejected under #136 after94.244s. No GPU launch. **LLVM21 `f7ceeef`:** 144 later pointer/i64 mismatches, 94.578 s; no GPU launch. Genuine pointer-plus-length lanes match #118; literal `{1,0}` must not be retyped as a device pointer without a sound guard proof. [LLVM7 report](cumetal-76-scaling-validation.md); [LLVM21 report](cumetal-134-validation.md). |
| [#31 — Ethereum self-tests](https://github.com/brandonros/vanity-miner-rs/issues/31) | [#133](https://github.com/Lulzx/cuda-metal/issues/133) pipeline preparation for both named versions. | #130/#138 clears the LLVM7 guard failure. **No pipeline-performance correction claimed.** | New follow-ups absent. | CPU 8/8 pass. **LLVM7 `0f98856` and LLVM21 `4a207e2`:** MSL/library creation pass; each normal invocation times out at 600 s, sampled in compute-pipeline creation before dispatch. Same API stage does not establish identical optimizer cause. [LLVM7 report](cumetal-130-validation.md); [LLVM21 report](cumetal-four-self-test-validation.md). |
| [#32 — P-256 public-key self-tests](https://github.com/brandonros/vanity-miner-rs/issues/32) | Original [#116](https://github.com/Lulzx/cuda-metal/issues/116); fresh helper-global scope [#123](https://github.com/Lulzx/cuda-metal/issues/123) is a **lead**; additional address-space error **unassigned**. | [#117](https://github.com/Lulzx/cuda-metal/pull/117); #123 has unpublished WIP. | #117 **yes**; #123 **no**. | **13 checks blocked by Apple Metal compilation:** undeclared `private$39`, plus a pointer cast missing an explicit address space. PTX references the global in HMAC/scalar/end-to-end helpers. Test both error families; registration issue #125 is not established by these compilation errors. |
| [#33 — P-256 signature self-tests](https://github.com/brandonros/vanity-miner-rs/issues/33) | Original [#120](https://github.com/Lulzx/cuda-metal/issues/120); fresh CFG error **unassigned**, [#83](https://github.com/Lulzx/cuda-metal/issues/83)/#120 are leads. | [#121](https://github.com/Lulzx/cuda-metal/pull/121) clears the original eight-block-depth case. **No demonstrated fresh-case fix.** | #121 **yes**. | **13 checks blocked by translation:** `%p243` undefined at `$L__BB8_1` inside outlined `p256_signature.low_s`. Reduce the new validity-guard path; this is not proof the original `%rs506` correction regressed. |
| [#34 — RSA modulus self-tests](https://github.com/brandonros/vanity-miner-rs/issues/34) | Original [cuda-metal #35](https://github.com/Lulzx/cuda-metal/issues/35) → [#124](https://github.com/Lulzx/cuda-metal/issues/124), with optimization [#127](https://github.com/Lulzx/cuda-metal/issues/127). Fresh masked-payload error: **#35 scope lead**, coverage unconfirmed. | [#122](https://github.com/Lulzx/cuda-metal/pull/122) clears the original masked-payload SSA. **No resource/PRMT fix PR or demonstrated new-case fix.** | #122 **yes**. | **16 checks blocked by translation:** `%rs907` undefined at `$L__BB16_1` inside new `rsa_modulus.range_multiple`. The comparison is masked by a presence predicate; reduce it before extending #35 acceptance. This run never reaches the original artifact's Metal allocation failure. |
| [#35 — RSA-PSS self-tests](https://github.com/brandonros/vanity-miner-rs/issues/35) | Current LLVM7: [#141 — direct CLZ](https://github.com/Lulzx/cuda-metal/issues/141). Original [#118 — mixed lanes](https://github.com/Lulzx/cuda-metal/issues/118) retains its separate historical reproducer. | **#142 clears LLVM7 address-demand scaling. No #141/#118 fix claimed.** | New follow-ups absent from historical `9e3e615`. | **`fb3644a`:** normal LLVM7 replay rejects `clz.b64` at55734 after80.859s;14 checks blocked, zero GPU launches. CPU14/14 pass. LLVM21 retains historical `c4e5fac` CLZ rejection. Older Metal/global diagnostics are historical. [Report](cumetal-76-scaling-validation.md). |
| [#37 — Shallenge self-tests](https://github.com/brandonros/vanity-miner-rs/issues/37) | [#118](https://github.com/Lulzx/cuda-metal/issues/118) heterogeneous vector pointer reloads. | #76/#135 and #130/#138 clear preceding memory/guard proof failures. **No #118 fix claimed.** | New follow-ups absent. | CPU 21/21 pass. **LLVM7 `13efc29`:** mixed-vector translation rejection. **LLVM21 `0f98856`:** MSL emitted, Apple compilation rejects pointer qualifiers/casts (16.890 s). Exact source chains recover private pointer/length fields through homogeneous vector inference; published on #118. [Report](cumetal-130-validation.md). |
| [#38 — LLVM 7 Shallenge production](https://github.com/brandonros/vanity-miner-rs/issues/38) | [#129 — proven same-base address cancellation](https://github.com/Lulzx/cuda-metal/issues/129) → [PR #132](https://github.com/Lulzx/cuda-metal/pull/132). | **Implemented and CPU/GPU verified at `4a207e2`; draft PR unmerged.** | No; added in `4a207e2`. | **Closed as completed on 2026-09-16.** Both unchanged LLVM versions pass all 33 profiles: 66 invocations, 132 GPU batches, 4,224 candidate positions and 96 independently verified winner hashes. [Measured report](cumetal-shallenge-validation.md). Self-test #37 is separate. |
| [#39 — P-256 public-key production](https://github.com/brandonros/vanity-miner-rs/issues/39), closed | [#130](https://github.com/Lulzx/cuda-metal/issues/130) → [PR #138](https://github.com/Lulzx/cuda-metal/pull/138). | **Implemented and normal-consumer CPU/GPU verified at `0f98856`; draft unmerged.** | No; newer implementation. | **Closed after 11 passing invocations, 22 GPU batches, 704 checked candidate positions, 18 independently reconstructed winners.** Original LLVM7 and both latest versions pass; five profiles per latest version cover all targets and selective/all/no-match cases. [Report](cumetal-130-validation.md). |


The two historically passing production rows are in the [status report](cumetal-status.md):
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

## Three-mode production acceptance complete; remaining work is separate

The [three-mode validation](cumetal-three-mode-validation.md) now verifies all
six production module/version combinations at `0242f22`. #131's nested-pointer
regression is corrected and numerically tested. The longer limit allowed all
first invocations to complete; startup optimization remains worthwhile but is
not an outstanding correctness requirement for this measured acceptance.

Continue the separately tracked self-test/proof cases, or profile
[#127](https://github.com/Lulzx/cuda-metal/issues/127) for faster startup. The
static inventory below remains useful because the PTX inputs are unchanged;
it does not establish byte shuffles as Apple's dominant compilation cost.

### Why #127 is a concrete candidate

Static inspection of the exact fresh PTX hashes finds immediate-selector byte
shuffles in all six production inputs:

| Production module | LLVM21 immediate / total `prmt.b32` | LLVM7 immediate / total `prmt.b32` |
| --- | ---: | ---: |
| Solana | 105 / 105 | 64 / 112 |
| Bitcoin | 168 / 168 | 105 / 131 |
| Ethereum | 116 / 116 | 42 / 58 |

Local inventory outside this repository:
`upstream-issue-breakdown/issue-76-implementation/issue127-production-prmt-inventory.json`.
These are static PTX counts, not emitted clone counts or measured speedups. The historical `c4e5fac`
LLVM21 outputs are 4,364,342 / 8,464,826 / 7,552,387 bytes respectively; size alone
neither proves the timeout cause nor predicts a successful optimization.

[#126 — explicit function preparation](https://github.com/Lulzx/cuda-metal/issues/126)
remains useful API work, but adding `cuFuncLoad` does not by itself reduce the
compilation cost of the preparation already triggered here. Keep #126 separate
from the performance investigation.

**Production acceptance:** empty address patterns are insufficient because these
three kernels return only the private key for a matching candidate. Require
selective nonempty prefix/suffix fixtures with known matching and nonmatching
candidates, CPU-verified match/error counts and returned winners, and independent
CPU reconstruction of each winner's address. Check unchanged inputs and buffer
guards across at least two batches to exercise reuse. Record the selected source,
paired compiler/runtime and PTX identities with each result. The new three-mode
run at `0242f22` satisfies these bounded gates for both LLVM7 and LLVM21.

No RSA, P-256 or Shallenge issue is currently a demonstrated dependency for these
three production workloads.

## Shared fixes and next assignments

- **Shallenge production #38 verified at `4a207e2`:** #129 has implemented,
  published and numerically verified cancellation in draft PR #132. Its upstream
  integration is pending; Shallenge self-test #37 remains independent.
- **Production #23–25 verified at `0242f22`:** #131's core type correction and
  qualifier follow-up now pass both LLVM versions through actual CPU/GPU acceptance.
  Keep #76's remaining proof cases and broader #118 per-lane pointer work separately
  tracked; these production passes do not establish their completion.
- **#130/#134 implementations are published:** PR #138 clears scalar-zero guard
  failures and verifies P-256 production; PR #139 clears unsigned pack/narrowing
  failures. Later self-test owners are #140/#136 for Solana, #76/#118 plus the
  empty-slice boundary for Bitcoin, #118 for Shallenge, and #133 for Ethereum.
  Use the named revisions above instead of repeating the old first diagnostics.
- **#76 follow-up is scoped:** original aligned-address and Shallenge range
  gates clear. Later proof exhaustion and native legacy numerical acceptance
  remain, with #136/#137 independently tracking helper memory effects.
- **Helper-global work:** keep #123 and #125 coordinated, while retaining the
  distinction between source generation, metadata discovery and GPU binding.
  Independently reduce the new Metal pointer-cast errors.
- **Resource investigations:** keep RSA-PSS library compilation (#115), historical
  RSA self-test allocation failure (#124), historical RSA GPU completion (#119),
  and the recorded RSA pipeline-creation timeout separate until shared causes are
  proved. #131 Bitcoin/Ethereum completed under the longer limit; their historical
  90-second cutoffs are not unresolved hangs. #127 is an optimization candidate;
  #128 supplies diagnostic spans.
- **One integration owner:** control the pin, compiler/runtime pairing, PTX
  identities and serialized GPU measurements. A proposed patch must state which
  exact reproducer clears and which next stage remains.

Historical full-sweep evidence is retained in
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
