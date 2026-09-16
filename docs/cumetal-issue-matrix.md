# CuMetal issue ownership and fix status

**Current pin: `13efc293c6cd06da5616a87b2dd8db81b1b0e0df`.**
[Draft PR #135](https://github.com/Lulzx/cuda-metal/pull/135) follows up
[#76](https://github.com/Lulzx/cuda-metal/issues/76), stacked on #132 at `4a207e2`.
The matched Nix compiler/runtime and normal miner were rebuilt. The
[three-combination replay](cumetal-76-followup-validation.md) confirms the next
blockers below, with 147 CPU checks passing and no GPU self-test completion.
Other results remain attributed to their measured revisions.

## Latest compiler follow-up: #76 proof gates implemented; workloads remain open

At `13efc29`, focused development builds pass **55 PTX functional tests each** in
Release/shim OFF and Debug/shim ON, including numerical address/memory and
conversion checks. GitGuardian passes on the exact published head. Native legacy
GPU acceptance still needs the offline Apple toolchain, so #76 remains open for
that validation, review and integration. [Acceptance details](https://github.com/brandonros/cuda-metal/blob/13efc293c6cd06da5616a87b2dd8db81b1b0e0df/docs/issue-76-acceptance-tests.md).

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
from these seven translation results. #130 implementation is next, followed by
#134. The separate Ethereum pipeline-preparation owner remains
[#133](https://github.com/Lulzx/cuda-metal/issues/133).

## Earlier pin: Shallenge production accepted at `4a207e2`

**Shallenge production now passes both LLVM7 and LLVM21. Downstream #38 is closed.**
Upstream #129 is **implemented and CPU/GPU verified, with its PR still unmerged**.
It remains open for integration, not because the measured workload still fails.
GitGuardian passed on the exact PR head. These results were published and read
back on PR #132, upstream #129 and downstream #38 on 2026-09-16.

## Latest result: four self-test groups remain blocked on both versions

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
The separate Shallenge self-tests (#37) fail translation on both versions in
the newer replay above and remain open.

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
**LLVM7 Solana self-tests and both Shallenge self-test inputs were rerun at
`13efc29`.** The remaining four-group results retain `4a207e2`; the other rows
retain their explicitly named historical revisions. A PR number alone does not
identify a tested build. There is no fresh current-pin 16-workload score.

The earlier full translation sweep measured LLVM7 **3/16** and LLVM21 **7/16**
at `c4e5fac`. Those totals are historical; no current-pin 16-workload score is claimed.

| Workload / downstream tracker | Translation revision | LLVM7 PTX → Metal | LLVM21 PTX → Metal | Apple preparation at named revision | GPU + CPU at named revision |
| --- | --- | --- | --- | --- | --- |
| Solana production ([#23](https://github.com/brandonros/vanity-miner-rs/issues/23)) | `0242f22` | **Pass at `0242f22`** | **Pass at `0242f22`** | **Both versions passed** | **Both versions passed all four profiles** |
| Bitcoin production ([#24](https://github.com/brandonros/vanity-miner-rs/issues/24)) | `0242f22` | **Pass at `0242f22`** | **Pass at `0242f22`** | **Both versions passed** | **Both versions passed all four profiles** |
| Ethereum production ([#25](https://github.com/brandonros/vanity-miner-rs/issues/25)) | `0242f22` | **Pass at `0242f22`** | **Pass at `0242f22`** | **Both versions passed** | **Both versions passed all four profiles** |
| Shallenge production ([#38](https://github.com/brandonros/vanity-miner-rs/issues/38), closed) | `4a207e2` | **Pass at `4a207e2`** | **Pass at `4a207e2`** | **Both versions passed** | **Both versions passed all 33 profiles**; 132 GPU batches / 4,224 positions total |
| P-256 public-key production ([#39 (LLVM7)](https://github.com/brandonros/vanity-miner-rs/issues/39)) | `c4e5fac` (historical) | Rejected: undefined edge | **MSL emitted** | Not rerun | Not rerun |
| P-256 signature production ([#26](https://github.com/brandonros/vanity-miner-rs/issues/26)) | `c4e5fac` (historical) | Rejected: bitwise-address join | Rejected: undefined edge | Not rerun | Not rerun |
| RSA modulus production ([#27](https://github.com/brandonros/vanity-miner-rs/issues/27)) | `c4e5fac` (historical) | Rejected: undefined edge | Rejected: `clz.b64` unsupported | Not rerun | Not rerun |
| RSA-PSS production ([#28](https://github.com/brandonros/vanity-miner-rs/issues/28)) | `c4e5fac` (historical) | Rejected: vector parameter transfer | Rejected: `clz.b64` unsupported | Not rerun | Not rerun |
| Solana self-tests ([#29](https://github.com/brandonros/vanity-miner-rs/issues/29)) | LLVM7 `13efc29`; LLVM21 `4a207e2` | Rejected: helper-write proof (#136) | Rejected: undefined `%rd42` (#130) | Not reached | CPU 79/79 pass; GPU blocked at named revisions |
| Bitcoin self-tests ([#30](https://github.com/brandonros/vanity-miner-rs/issues/30)) | `4a207e2` | Rejected: undefined `%rd29462` | Rejected: undefined `%r1688` | Not reached | CPU 39/39 pass; GPU blocked on both versions |
| Ethereum self-tests ([#31](https://github.com/brandonros/vanity-miner-rs/issues/31)) | `4a207e2` | Rejected: undefined `%rd29462` | **MSL emitted** | LLVM21 library creation passes; **pipeline creation times out at 600 s** | CPU 8/8 pass; GPU blocked on both versions |
| Shallenge self-tests ([#37](https://github.com/brandonros/vanity-miner-rs/issues/37)) | `13efc29` | Rejected: mixed-vector pointer (#118) | Rejected: zero-marker guard (#130) | Not reached | CPU 21/21 pass; GPU blocked on both versions |
| P-256 public-key self-tests ([#32](https://github.com/brandonros/vanity-miner-rs/issues/32)) | `c4e5fac` (historical) | Rejected: bitwise-address join | **MSL emitted** | Not rerun | Not rerun |
| P-256 signature self-tests ([#33](https://github.com/brandonros/vanity-miner-rs/issues/33)) | `c4e5fac` (historical) | Rejected: bitwise-address join | Rejected: undefined edge | Not rerun | Not rerun |
| RSA modulus self-tests ([#34](https://github.com/brandonros/vanity-miner-rs/issues/34)) | `c4e5fac` (historical) | Rejected: undefined edge | Rejected: `clz.b64` unsupported | Not rerun | Not rerun |
| RSA-PSS self-tests ([#35](https://github.com/brandonros/vanity-miner-rs/issues/35)) | `c4e5fac` (historical) | Rejected: bitwise-address join | Rejected: `clz.b64` unsupported | Not rerun | Not rerun |

## Next work and closure

1. **Implementation and measured acceptance complete: #129/#132; #38 closed.** The published implementation
   passes its scoped regression and both-version production gates. PR #132 is
   a draft because it depends on the pending #131 stack; upstream integration
   remains outstanding. This does not close the separate Shallenge self-tests.
2. **Four self-test groups freshly measured; eight open workloads still need
   current-pin reruns.** #29/#30/#31/#37 remain blocked as recorded above. #39 has the
   confirmed scalar-zero guard owner [#130](https://github.com/Lulzx/cuda-metal/issues/130).
   The last `c4e5fac` sweep also records five LLVM7 bitwise-address joins
   (#26/#29/#32/#33/#35) and Shallenge self-test memory-proof failures (#37).
   The new replay confirms #29's LLVM7 pointer join and both #37 memory-proof
   failures at `4a207e2`; #26/#32/#33/#35 were not retested. LLVM21 #31 now has
   a measured pipeline-preparation timeout owned by new #133; its cause/fix
   remains under investigation. The newly assigned #130/#134 work is separate
   from #76's proof gates.
3. **For faster startup, measure #127's optimization separately.**
   [#127](https://github.com/Lulzx/cuda-metal/issues/127) concerns constant byte
   shuffles; [PR #128](https://github.com/Lulzx/cuda-metal/pull/128) adds diagnostics.
   Static source size alone does not prove which Apple compiler pass dominates.

The last `c4e5fac` LLVM21 RSA #27/#28/#34/#35 attempts stopped at unsupported
`clz.b64`; that owner remains unassigned here. RSA was not rerun at the current
pin, and its historical resource cases remain unresolved. Ethereum self-tests
(#31) now emit LLVM21 MSL at `4a207e2` but time out in Apple pipeline preparation
before a GPU numerical result.

**Four workload issues are closed: #23–25 and #38.** The other 12 workload
trackers and broader #16/#19 remain open. Four of those 12 workloads were rerun
on `4a207e2` and remain blocked; the other eight were not rerun, so this is not
a claim that all still fail. Historical results must not
be combined with the current row into a fresh 16-row score. Other upstream
issues retain their own acceptance and integration gates.

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
| [#39 — LLVM 7 P-256 public-key production](https://github.com/brandonros/vanity-miner-rs/issues/39) | `p256_public_key.ptx`: exit 1 after 9.412 s; `%rd18761` undefined on an incoming edge to `$L__BB5_1`. SHA-256 `dbd5ada9e96613eb7f2da98e97b95310a5a6368744c5f8d1d84b9c184184e0da`. | LLVM 21 emits MSL in 6.906 s on this bundle. Production is distinct from self-test issue #32. [Upstream #130](https://github.com/Lulzx/cuda-metal/issues/130) now tracks the missing scalar zero-marker guard proof; a compact case distinguishes it from #83 empty-label and #120 depth failures. |

Both published issues retain producer `7484a5d` plus the recorded timing
instrumentation, CuMetal `9e3e615`, the exact compiler and input hashes, and the
full diagnostic. First require the retained LLVM 7 input to translate, then
complete Apple compilation and bounded CPU-verified production batches with
input/buffer guards intact; retain an independent LLVM 21 regression row. A
compiler correction remains unimplemented: focused research and complete small
reproducers are published in upstream **#129** and **#130**, linked from the
downstream issues. These assignments cover the demonstrated first gaps, not
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
supported; research leads are labeled explicitly. No blocked self-test row is
an individual numerical assertion failure.

| Vanity-miner issue | Upstream ownership | Published implementation | Included in `9e3e615`? | Recorded result and latest next work |
| --- | --- | --- | --- | --- |
| [#23 — Solana production](https://github.com/brandonros/vanity-miner-rs/issues/23) | [#76 / #131](https://github.com/Lulzx/cuda-metal/issues/76); [#113](https://github.com/Lulzx/cuda-metal/issues/113) pointer-load acceptance | **Implemented in `0242f22`; both LLVM versions CPU/GPU verified.** [Measured report](cumetal-three-mode-validation.md). | Earlier PRs only; #131 was not in historical `9e3e615`. | **`0242f22`:** eight invocations / 16 GPU batches across both versions pass CPU counts/payload, guards and independent-address checks. **Closed as completed on 2026-09-16** after publishing and verifying this evidence. |
| [#24 — Bitcoin production](https://github.com/brandonros/vanity-miner-rs/issues/24) | [#76 / #131](https://github.com/Lulzx/cuda-metal/issues/76); [#113](https://github.com/Lulzx/cuda-metal/issues/113) pointer-load acceptance | **Implemented in `0242f22`; both LLVM versions CPU/GPU verified.** [Measured report](cumetal-three-mode-validation.md). | Earlier PRs only; #131 was not in historical `9e3e615`. | **`0242f22`:** eight invocations / 16 GPU batches across both versions pass CPU counts/payload, guards and independent-address checks. **Closed as completed on 2026-09-16** after publishing and verifying this evidence. |
| [#25 — Ethereum production](https://github.com/brandonros/vanity-miner-rs/issues/25) | [#76 / #131](https://github.com/Lulzx/cuda-metal/issues/76); [#113](https://github.com/Lulzx/cuda-metal/issues/113) pointer-load acceptance | **Implemented in `0242f22`; both LLVM versions CPU/GPU verified.** [Measured report](cumetal-three-mode-validation.md). | Earlier PRs only; #131 was not in historical `9e3e615`. | **`0242f22`:** eight invocations / 16 GPU batches across both versions pass CPU counts/payload, guards and independent-address checks. **Closed as completed on 2026-09-16** after publishing and verifying this evidence. |
| [#26 — P-256 signature production](https://github.com/brandonros/vanity-miner-rs/issues/26) | [#83 — empty labels/predicate facts](https://github.com/Lulzx/cuda-metal/issues/83) | #85 covers original predicate cases; **no empty-label follow-up PR**. | #85 and #121 included; #121 addresses a different depth case. | **Both sources fail PTX translation:** `%rd17653` undefined at `$L__BB0_11`. Preserve facts across empty blocks and validate message/ephemeral search. |
| [#27 — RSA modulus production](https://github.com/brandonros/vanity-miner-rs/issues/27) | [#119](https://github.com/Lulzx/cuda-metal/issues/119) owns the **historical GPU-completion wait**. Fresh pipeline-preparation wait: **owner unconfirmed**. | **No completion/performance fix PR.** | No corresponding fix. | **Times out:** fresh samples at about 180 s and near the deadline are inside **Metal pipeline creation**, before verified execution. This is not the historical device-synchronization sample. Attribute the preparation cost and confirm an owner; then complete verified four-stage cycles. |
| [#28 — RSA-PSS production](https://github.com/brandonros/vanity-miner-rs/issues/28) | Earlier #96/#109/#111 → [#115 — Metal compilation time](https://github.com/Lulzx/cuda-metal/issues/115) | #108/#110/#112 clear translation; [#128](https://github.com/Lulzx/cuda-metal/pull/128) adds **diagnostics only**. **No performance fix PR.** | Translation fixes yes; #128 **no**. | **Salt and message searches time out** while sampled in Metal source-library compilation. Both emit source; neither verifies candidates. Diagnose and reduce compilation cost, then validate both sources. |
| [#29 — Solana self-tests](https://github.com/brandonros/vanity-miner-rs/issues/29) | **LLVM7: [#76](https://github.com/Lulzx/cuda-metal/issues/76). LLVM21: [#130](https://github.com/Lulzx/cuda-metal/issues/130).** Original #46/#116 trap scope is historical. | #117 clears the original trap helper. #131 retains the LLVM7 bitwise-address proof as open; **#130 zero-sentinel correction remains unimplemented**, with current reducers published. | #117 **yes**. | **`4a207e2`, producer `afe80210`:** CPU 79/79 pass. LLVM7 rejects pointer/null branch argument `%rd785` at `$L__BB4_3`; LLVM21 rejects undefined `%rd42` at `$L__BB77_6`. Both block before GPU execution. Published reductions assign the LLVM21 guard to #130; keep #29 open. [Report](cumetal-four-self-test-validation.md). |
| [#30 — Bitcoin self-tests](https://github.com/brandonros/vanity-miner-rs/issues/30) | **LLVM7: [#130](https://github.com/Lulzx/cuda-metal/issues/130). LLVM21: new [#134](https://github.com/Lulzx/cuda-metal/issues/134).** #51/#66 tuple-extraction scopes do not cover this unsigned-narrowing consumer. | Historical #114/#117 fixes remain; **#130/#134 required corrections are not implemented in this pin**. Current small controls and acceptance are published. | **Yes**, existing scoped fixes. | **`4a207e2`, producer `afe80210`:** CPU 39/39 pass. LLVM7 rejects undefined `%rd29462` at `$L__BB48_1`; LLVM21 rejects undefined `%r1688` at `$L__BB4_3`. Both block before GPU execution. Guard proof is assigned to #130 and unsigned-narrowing demand to #134; keep #30 open. [Report](cumetal-four-self-test-validation.md). |
| [#31 — Ethereum self-tests](https://github.com/brandonros/vanity-miner-rs/issues/31) | **LLVM7: [#130](https://github.com/Lulzx/cuda-metal/issues/130). LLVM21 preparation: new [#133](https://github.com/Lulzx/cuda-metal/issues/133).** #76/#131 clear the measured LLVM21 translation stage. | #117 covers the old trap helper; #131 emits LLVM21 MSL. **#130 guard correction unimplemented; #133 preparation cause/fix under investigation.** | #117 yes; **#131 no**. | **`4a207e2`, producer `afe80210`:** CPU 8/8 pass. LLVM7 rejects undefined `%rd29462` at `$L__BB15_1`. LLVM21 emits MSL and completes source-library creation/function lookup, then **times out at 600 s in pipeline preparation** (samples at about 120/578 s). No GPU result; performance owner #133; cause/fix remain unproven. Keep #31 open. [Report](cumetal-four-self-test-validation.md). |
| [#32 — P-256 public-key self-tests](https://github.com/brandonros/vanity-miner-rs/issues/32) | Original [#116](https://github.com/Lulzx/cuda-metal/issues/116); fresh helper-global scope [#123](https://github.com/Lulzx/cuda-metal/issues/123) is a **lead**; additional address-space error **unassigned**. | [#117](https://github.com/Lulzx/cuda-metal/pull/117); #123 has unpublished WIP. | #117 **yes**; #123 **no**. | **13 checks blocked by Apple Metal compilation:** undeclared `private$39`, plus a pointer cast missing an explicit address space. PTX references the global in HMAC/scalar/end-to-end helpers. Test both error families; registration issue #125 is not established by these compilation errors. |
| [#33 — P-256 signature self-tests](https://github.com/brandonros/vanity-miner-rs/issues/33) | Original [#120](https://github.com/Lulzx/cuda-metal/issues/120); fresh CFG error **unassigned**, [#83](https://github.com/Lulzx/cuda-metal/issues/83)/#120 are leads. | [#121](https://github.com/Lulzx/cuda-metal/pull/121) clears the original eight-block-depth case. **No demonstrated fresh-case fix.** | #121 **yes**. | **13 checks blocked by translation:** `%p243` undefined at `$L__BB8_1` inside outlined `p256_signature.low_s`. Reduce the new validity-guard path; this is not proof the original `%rs506` correction regressed. |
| [#34 — RSA modulus self-tests](https://github.com/brandonros/vanity-miner-rs/issues/34) | Original [cuda-metal #35](https://github.com/Lulzx/cuda-metal/issues/35) → [#124](https://github.com/Lulzx/cuda-metal/issues/124), with optimization [#127](https://github.com/Lulzx/cuda-metal/issues/127). Fresh masked-payload error: **#35 scope lead**, coverage unconfirmed. | [#122](https://github.com/Lulzx/cuda-metal/pull/122) clears the original masked-payload SSA. **No resource/PRMT fix PR or demonstrated new-case fix.** | #122 **yes**. | **16 checks blocked by translation:** `%rs907` undefined at `$L__BB16_1` inside new `rsa_modulus.range_multiple`. The comparison is masked by a presence predicate; reduce it before extending #35 acceptance. This run never reaches the original artifact's Metal allocation failure. |
| [#35 — RSA-PSS self-tests](https://github.com/brandonros/vanity-miner-rs/issues/35) | Original [#118 — mixed pointer lanes](https://github.com/Lulzx/cuda-metal/issues/118) remains open. Fresh pointer-cast error **unassigned**; later raw-global expressions are a [#123](https://github.com/Lulzx/cuda-metal/issues/123)-adjacent **lead**. | **No PR for #118; no demonstrated fix for the fresh Metal errors.** #108/#114/#117 cover other cases. | Related fixes yes; missing work **no**. | **14 checks blocked by Apple Metal compilation:** invalid `as_type<device uchar*>` from `ulong`, then syntax errors from unlowered `[private$em]` expressions. The latter symbol occurs in CRT helpers. Changed PTX reaching Metal does not demonstrate #118 fixed on its original reproducer. |
| [#37 — Shallenge self-tests](https://github.com/brandonros/vanity-miner-rs/issues/37) | **Both current memory-proof cases are already tracked in [#76](https://github.com/Lulzx/cuda-metal/issues/76)** and explicitly retained by [PR #131](https://github.com/Lulzx/cuda-metal/pull/131). Historical #83 predicate lead does not describe these current first failures. | #131 records both cases as **remaining branch/iterator-range and alias-proof work**; no demonstrated fix for them. | No identified correction. | **`4a207e2`, producer `afe80210`:** CPU 21/21 pass. LLVM7 rejects a potentially overlapping local store at PTX line 3048; LLVM21 rejects an unresolved local store offset at line 9526 (store line 7377). Both block translation; these memory-proof cases remain open acceptance under #76/#131. Keep #37 open. Production #38 remains separately verified/closed. [Report](cumetal-four-self-test-validation.md). |
| [#38 — LLVM 7 Shallenge production](https://github.com/brandonros/vanity-miner-rs/issues/38) | [#129 — proven same-base address cancellation](https://github.com/Lulzx/cuda-metal/issues/129) → [PR #132](https://github.com/Lulzx/cuda-metal/pull/132). | **Implemented and CPU/GPU verified at `4a207e2`; draft PR unmerged.** | No; added in `4a207e2`. | **Closed as completed on 2026-09-16.** Both unchanged LLVM versions pass all 33 profiles: 66 invocations, 132 GPU batches, 4,224 candidate positions and 96 independently verified winner hashes. [Measured report](cumetal-shallenge-validation.md). Self-test #37 is separate. |
| [#39 — LLVM 7 P-256 public-key production](https://github.com/brandonros/vanity-miner-rs/issues/39) | [#130 — scalar zero-sentinel guard propagation](https://github.com/Lulzx/cuda-metal/issues/130). Distinct compact case from #83 empty labels and #120 depth. | **Research/reproducer published; no fix PR.** | No implementation. | **Original split-bundle input:** `%rd18761` undefined at `$L__BB5_1` after 9.412 s. Signed/unsigned small zero-marker cases fail; direct-predicate control emits. Full-input redundant guard controls advance to `%rd18765`, then `%rd18781`; the full module still fails. |


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

[Exact-input inventory](../../upstream-issue-breakdown/issue-76-implementation/issue127-production-prmt-inventory.json).
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
- **Four self-test ownership assignments are published:** #130 owns LLVM21
  Solana and LLVM7 Bitcoin/Ethereum; #134 owns LLVM21 Bitcoin narrowing; #133
  owns LLVM21 Ethereum preparation. #76 retains LLVM7 Solana and both Shallenge
  proofs. P-256/RSA's other undefined-value cases still need independent
  reductions; these assignments do not establish that all CFG errors are one defect.
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
