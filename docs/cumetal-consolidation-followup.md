# CuMetal consolidation and root-cause follow-up

Executed **2026-09-16 UTC** after the [64-issue scope audit](cumetal-issue-duplicate-audit.md).
GitHub body changes, the closure, and all four research comments were read back
and verified. Source investigation used immutable compiler `9e3e615`, not an
active contribution worktree. No compiler implementation or dependency pin changed.

## Published tracking changes

| Tracker | Applied change |
| --- | --- |
| [#46](https://github.com/Lulzx/cuda-metal/issues/46) | Closed with reason `not_planned`, explicitly superseded by #116. Historical body preserved. |
| [#116](https://github.com/Lulzx/cuda-metal/issues/116) | Canonical owner; received the original four artifact hashes and reconciled helper value/effect/cancellation acceptance before #46 was closed. Remains open. |
| [PR #117](https://github.com/Lulzx/cuda-metal/pull/117) | References #46's superseded plan and carries unchecked review/verification gates for the transferred acceptance. Remains open. |
| [#83](https://github.com/Lulzx/cuda-metal/issues/83) | Current body owns empty-label handling and retains #85 regressions. Its historical depth subcase is reassigned, not repeated as missing work. Remains open. |
| [#120](https://github.com/Lulzx/cuda-metal/issues/120) | Explicitly owns the historical depth failure through #121; cross-links the distinct empty-label scope. Remains open. |

This removes one tracker from the original audited set: **63 of those 64 remain
open**. It does not establish 63 distinct bugs or mark any GPU workload complete.
The obsolete finite-loop classifier and mandatory expansion plan were replaced
with reusable guarded helpers, shared status and cancellation. Full workload
acceptance remains in the downstream issues. Original-input translation and
focused numerical/cancellation tests remain explicit upstream gates.

## #80: missing generic-to-global helper pointer recovery

[Published reproducer, controls and source attribution](https://github.com/Lulzx/cuda-metal/issues/80#issuecomment-5691940830).


Both unchanged historical inputs reproduce `address-space casts require explicit
pointer address spaces` on the pinned compiler:

| Input/control | Translation result | Elapsed |
| --- | --- | --- |
| Original guarded `ffd92adb…` | Same cast failure | 56.904 s |
| Original unguarded `4e08de09…` | Same cast failure | 59.343 s |
| Unguarded helper: explicit `.ptr` in forward declaration and definition | Emits MSL | 54.950 s on retry; initial 60 s attempt timed out |
| Unguarded helper: replace exactly one `cvta.to.global` with `mov.b64` | Emits MSL | 53.446 s |

A complete 31-line synthetic probe reproduces the same error with no branches,
joins, traps or promoted globals. Adding `.ptr`, replacing the conversion with a
move, eliminating the offset, or using a literal offset emits MSL. Commuting the
base and dynamic-index operands still fails. Small probes took 0.012–0.023 s;
these single invocations are diagnostic observations, not benchmarks.

The parser loses single-parameter provenance when base and index both come from
different parameters. Typed backward recovery then handles the addition but
stops at `cvta.to.global`: it explicitly seeds `cvta.to.local`, not this global
form, and its propagation does not follow `cvta`. The source remains scalar, so
conversion lowering creates the rejected cast. Explicit pointer/move controls
supply the missing evidence without requiring a join fix.

**Retain #80 as a distinct extension beyond #39/#56's local-helper recovery.**
#67's existing promoted-global correction and #76's join work do not subsume this
straight-line acceptance. The fix should infer pointer-ness while retaining
generic provenance until call-site specialization; forcing device storage would
be wrong for the original helper's promoted immutable-table caller. Preserve
invalid-conversion rejection and scalar index types.

No compiler fix was implemented. Diagnostic PTX changes are not production
workarounds. Emission concerns the selected `kernel_repro_alphabet_helper` entry,
not every entry in these large files. Apple compilation, table/device-buffer
outputs and GPU guards remain unverified.

## #115/#124: shared avoidable PRMT work, resource causes still unresolved

Published on [#115](https://github.com/Lulzx/cuda-metal/issues/115#issuecomment-5691941301), [#124](https://github.com/Lulzx/cuda-metal/issues/124#issuecomment-5691941738) and [optimization owner #127](https://github.com/Lulzx/cuda-metal/issues/127#issuecomment-5691942186).


| Measurement | #115 RSA-PSS production | #124 RSA-modulus self-test |
| --- | --- | --- |
| Original MSL bytes | 24,263,894 | 53,578,468 |
| Entry dispatcher cases | 5,956 | 11,046 |
| Entry value declarations | 286,595 | 593,098 |
| Complete-module matched PRMT sequences | 878 | 4,461 |
| Entry-only matched sequences | 846 | 4,376 |
| Diagnostic specialized MSL bytes | 21,059,008 | 36,500,956 |
| Source reduction | **13.21%** | **31.87%** |

The two producer compiler revisions (`e5acf8cc0c65` and `9e3e615`) have a
byte-identical `ptx_bit_ops.cpp`. The matcher verifies all 66 statements and their
dependencies against that template. The diagnostic retains the two input
conversions, substitutes direct byte selection into the final result, and
removes 63 intermediates only after proving they have no escaping uses. All
unmatched lines and non-PRMT effects remain in the diagnostic outputs; originals
are preserved. These are source measurements, not Apple IR counts or live ranges.

CPU-only expression comparison passed **984,000 cases**, covering observed
selectors, sign replication, high selector bits, unsigned boundaries and
pseudorandom inputs. This checks the generated expressions, not complete source
semantic equivalence or GPU correctness. These separate diagnostic outputs are
not an implemented CuMetal optimization.

**#127 is a shared optimization target; #115/#124 are not established duplicates.**
No new Metal library/pipeline compilation was run, so neither the timeout nor the
allocation failure is demonstrated fixed or attributed to PRMT alone. The offline
`xcrun metal` utility was unavailable; this does not imply runtime Metal
compilation is unavailable. No runtime compilation result is claimed.

The next deciding experiment is a tested IR-lowering correction, followed by
re-emission of both unchanged original PTX files and controlled measurements of
library compilation and pipeline creation separately. Retain exact identities,
cache/load conditions, declared time/memory limits, error records and independent
workload acceptance. A source-size reduction alone is not that experiment.

## Evidence

Compiler used for #80: `9e3e61574b776424a96c686bdbdc04ad1f27fe9f`, SHA-256
`5662a763e6e63539cd9776efb844187c707d23297c35d2950c1e5820eb9ea628`.
No GPU runtime was loaded by these research commands.

| Artifact | SHA-256 |
| --- | --- |
| #80 guarded original PTX | `ffd92adbf11db672294ba6a7086bfe8a9aaa0438b1c6ac4827fde1fc7d323af3` |
| #80 unguarded original PTX | `4e08de091993f2475cd12ee8ecb13620ead3683ec96a3e2f52c2f231416b4b12` |
| #115 original MSL | `cd2801d8ea1b11b116002c6bc8f4e991a3667c0699274ca36da1e0f93811f196` |
| #115 diagnostic MSL | `1cddffd6988fb11511463b658145bd558405660631ca50a5d6073e652c4461d7` |
| #124 original MSL | `1af1c4456116dab606bfc5274ee9d90d4d731666eefa9f212ec82a933a359766` |
| #124 diagnostic MSL | `0e28ce25526d247ea53fd8cd3f4163bafa2845bb3613807673a564a69df2fcff` |

Local retained evidence: `.cumetal-artifacts/consolidation-20260916/` contains
before/after GitHub bodies, publication receipts, exact compiler commands,
original-input references and hashes, small probes, source matcher/diagnostics,
CPU expression checker and JSON results. It is ignored local evidence, not a
public attachment. The linked GitHub comments contain the complete #80 reducer,
controls, implementation locations, resource-comparison method and limits.
