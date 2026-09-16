# Four self-test groups on CuMetal `4a207e2`

**Measured 2026-09-16. None of #29, #30, #31 or #37 meets closure criteria.**
All four groups were attempted under both LLVM7 and LLVM21 using the current
locked Nix compiler/runtime package and freshly built normal CLI runners.

- **CPU: 147/147 checks pass.**
- **CuMetal: 0/8 module/version combinations pass.** Seven fail PTX translation;
  LLVM21 Ethereum reaches Metal pipeline creation and exceeds its 600-second deadline.
- **294 GPU check selections are blocked:** 286 by translation, eight by the
  preparation timeout. There are zero GPU passes, numerical failures or skips.
  The timeout produces no named outcomes; all eight affected selections are accounted for.
- No GPU result readback or guard validation completed. Existing production
  acceptance and closed production issues are unchanged.

## Results

| Self-test group / issue | LLVM7 | Seconds | LLVM21 | Seconds | Close? |
| --- | --- | ---: | --- | ---: | --- |
| Solana [#29](https://github.com/brandonros/vanity-miner-rs/issues/29) | Translation blocked: pointer/null branch argument `%rd785` at `$L__BB4_3`. | 85.707 | Translation blocked: `%rd42` undefined at `$L__BB77_6`. | 69.508 | **No** |
| Bitcoin [#30](https://github.com/brandonros/vanity-miner-rs/issues/30) | Translation blocked: `%rd29462` undefined at `$L__BB48_1`. | 84.397 | Translation blocked: `%r1688` undefined at `$L__BB4_3`. | 62.814 | **No** |
| Ethereum [#31](https://github.com/brandonros/vanity-miner-rs/issues/31) | Translation blocked: `%rd29462` undefined at `$L__BB15_1`. | 18.888 | Metal source-library creation passed; compute-pipeline preparation timed out at 600 seconds. | 600.014 | **No** |
| Shallenge [#37](https://github.com/brandonros/vanity-miner-rs/issues/37) | Translation blocked: unresolved potentially overlapping local store (PTX line 3048). | 17.785 | Translation blocked: unresolved local store offset (PTX line 9526; store at line 7377). | 9.991 | **No** |

Times are whole CLI invocations, including translation and any later preparation.
A translation failure is one blocked module, even though the CLI reports it
against every selected check. Similar undefined-register diagnostics do not
establish that the workloads share one compiler defect.

## Published upstream ownership

**All eight measured mode/version failures now have explicit upstream issue
owners.** Published and read back on 2026-09-16: two new issues, expanded
acceptance on #130, the current #76 checklist, and current ownership at the top
of downstream #29/#30/#31/#37. All four downstream tickets remain open.

| Group | LLVM7 current blocker | LLVM21 current blocker |
| --- | --- | --- |
| Solana #29 | [#76](https://github.com/Lulzx/cuda-metal/issues/76): remaining bitwise-address join proof | [#130](https://github.com/Lulzx/cuda-metal/issues/130): scalar-zero guard with two skipped-definition edges |
| Bitcoin #30 | [#130](https://github.com/Lulzx/cuda-metal/issues/130): copied scalar zero sentinel in the shared k256 helper | **New [#134](https://github.com/Lulzx/cuda-metal/issues/134):** unsigned narrowing of a tuple with an unobserved undefined high half |
| Ethereum #31 | [#130](https://github.com/Lulzx/cuda-metal/issues/130): same guarded k256 payload as Bitcoin | **New [#133](https://github.com/Lulzx/cuda-metal/issues/133):** compute-pipeline preparation timeout; cause/fix still under investigation |
| Shallenge #37 | [#76](https://github.com/Lulzx/cuda-metal/issues/76): local-store alias/range proof | [#76](https://github.com/Lulzx/cuda-metal/issues/76): local-store alias/range proof |

### Research establishing these owners

The audit checked the existing 76 issue bodies, 45 comments and published PR
scopes before creating #133/#134. Twenty-five tiny compiler-only probes used
the exact `4a207e2` compiler: six shared-k256 guard probes, seven Solana guard
probes, and twelve Bitcoin pack/narrow probes. No compiler source, pin, PTX
acceptance input, or GPU result was changed by this research.

- [#130's published reductions](https://github.com/Lulzx/cuda-metal/issues/130#issuecomment-5702778539)
  show that LLVM7 Bitcoin/Ethereum share the same complete normalized k256
  helper and scalar-zero guard. LLVM21 Solana uses the same existing issue's
  dynamic-marker proof with two early exits. Predicate controls emit MSL;
  observable undefined-value controls remain rejected. The original P-256
  public-key production #39 acceptance stays with #130.
- #134 includes a complete loop and straight-line reduction: unsigned narrowing
  discards the undefined high32 lane, but SSA still demands it. Existing #51/#66
  explicitly cover tuple-extraction consumers; their controls emit on this pin.
  The new `cvt` consumer therefore receives its own bounded follow-up owner.
- #133 retains the exact Ethereum PTX/MSL identities, samples, public Actions
  input, reproduction and completion gates. #115's source-library wait,
  #124's allocation failure and #119's GPU-completion wait remain separate;
  a common cause has not been established. #127 is an optimization lead.

Tracking a first failure does not mean a fix exists or a whole workload passes.
All these measured gaps remain unresolved. Any later independent failure must
receive its own attribution after the current blocker clears.

Local [publication records](../../upstream-issue-breakdown/four-selftest-ownership/publication/published-bodies.json),
[shared guard research](../../upstream-issue-breakdown/four-selftest-ownership/llvm7-secp/findings.md),
[Solana guard research](../../upstream-issue-breakdown/four-selftest-ownership/solana-llvm21/findings.md),
and [Bitcoin probe results](../../upstream-issue-breakdown/four-selftest-ownership/bitcoin-llvm21/results.json)
retain the investigation. Public issues contain the small inputs and key evidence;
local evidence files are not public attachments.

### What remains on #76

1. **Implement proven bitwise-address legalization** for the five recorded LLVM7
   pointer-join cases. Prove alignment/low-bit conditions and preserve provenance
   through copies, selects and loops; keep unproven integer-to-pointer uses rejected.
2. **Implement local memory range/alias proof** for both Shallenge self-test inputs.
   Use actual definitions and valid guard/iterator bounds; retain conservative
   rejection of overlapping or unknown stores.
3. **Complete numerical/regression acceptance:** positive/negative tests and
   unchanged full-input replays for both corrections, plus the original
   both-backend joined-ReLU gates (legacy CFG variants are still untested).
4. **Complete proof-budget and conversion-boundary coverage:** exercise bounded
   fallback, and resolve exceptional-float/wider-container limits through tested
   supported behavior or explicit unsupported contracts.

PR #131's conversion correction and qualifier follow-up already have both-version
Solana/Bitcoin/Ethereum production acceptance. Those gates do not need to be
called unfinished again. The remaining proof work and upstream PR integration
are still open; #130/#133/#134 own the separately attributed failures above.

## Ethereum LLVM21: where the time went

The compiler emitted **12,415,178 bytes** of Metal source. Stack samples at
approximately **120 and 578 seconds** both show
`load_pipeline_locked` → `newComputePipelineStateWithFunction` → Apple compiler
service wait. In the pinned runtime, source-library creation and function lookup
must succeed before this call. The observed wait is therefore compute-pipeline
compilation/creation, before dispatch; it is not measured GPU execution time.

The supervisor terminated the owned process at the 600-second deadline
(600.014 seconds elapsed; exit `-15`). This establishes a bounded timeout,
not that compilation could never finish or which compiler optimization is costly.
The generated source, ABI sidecar and both stack samples are retained locally.

## Coverage and acceptance

| Group | Named checks per version | CPU passed | GPU selections across both versions |
| --- | ---: | ---: | ---: |
| Solana | 79 | 79 | 158 |
| Bitcoin | 39 | 39 | 78 |
| Ethereum | 8 | 8 | 16 |
| Shallenge | 21 | 21 | 42 |
| **Total** | **147** | **147** | **294** |

Both runners expose the same complete inventory. Every group selects every
owned `MODE.CHECK` name explicitly. Normal CuMetal self-test execution launches
one block with one thread, checks 16-byte boundary guards, verifies that
unrelated result slots retain their sentinel, and requires each selected result
to equal one. None of these four groups permits GPU skips. These acceptance
checks were not reached in this run.

An independent read-only audit verified all 12 command records, exact named
outcomes and summaries, timeout accounting, producer archive membership,
artifact hashes and logged toolchain/PTX identities. No discrepancy was found.

## Source and artifact identity

- Miner host: `80699f24bd477b5551bed1d8eff3bc910a833210`.
- PTX producer: `afe80210ea28748cc58c3ba75f877bfa4a6b1ecd`; [Actions run 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
- CuMetal: `4a207e277249370573d7fe170bb7b3346d9da015`.
- Matched Nix package: `/nix/store/is7s91d1vjk6ja2i3ldb4ijks3ccfygc-vanity-cumetal-4a207e277249`.
- Compiler SHA-256: `365e522b8c184233ec2fd1324cfee9b397adbfad121d90e1a99ab88e714c49a4`.
- Runtime SHA-256: `b3f1b7cd5ae3517cab97632b6b73a2a664930b6930334b7c5be02ab31c4b510d`.
- CPU CLI SHA-256: `cbeb9593afaf1c257915c6ca7f83b4326b045b366f20269b46031aed9dd528f7`.
- CuMetal CLI SHA-256: `9e956ec1bb7b4d704ad45ab9e67829bd413b900003d029bd9b1256a52004de74`.

Relevant CLI, logic, kernel, Cargo and PTX build sources match the PTX producer.
The newer dependency pin is deliberate. All eight PTX files match the retained
Actions artifact manifests and archive members. No PTX or compiler code was
modified for this replay. Workload specialization was disabled.

| Module | LLVM7 PTX SHA-256 | LLVM21 PTX SHA-256 |
| --- | --- | --- |
| `self_test_solana` | `d33cba71e185acd397a245f9b95908abc1c2f5d694d6ebb551ecbf4d4aef9344` | `71f679b22646b982d9954f7c5743e3880f4bd2166b0b2dbdcae138a5df86ccf3` |
| `self_test_bitcoin` | `0bbd7000a8949ff1ca3776c938c9f72f14da477f388a13983bb8e5cce4bb25bc` | `48a78769d28dc0e699b2e3209b332e8cd6efa5d3846cf39d1b9b0776adbe035b` |
| `self_test_ethereum` | `f15b78a5be18586821c4e22bbf6e56f3e58d913ef398894d732720a7772a7b02` | `91d19ff81140beb5d52d5a062b150daa4f23a63c6cbb7bb384f9bdaec34c936b` |
| `self_test_shallenge` | `a2e120bb765bfc5b90bea18ae2cf964843ea3a25e39d171316cae545f71c5f3b` | `de774d6d87edec30418f1ec76eb0499a635e159e73686e00b97d6df21ecf098c` |

## Reproduction and retained evidence

Build with `nix develop .#cumetal --command cargo build --offline --release
--locked -p vanity-miner --no-default-features`, enabling
`self_test_solana,self_test_bitcoin,self_test_ethereum,self_test_shallenge`.
Add `cumetal` for the GPU runner; use separate target directories for the two
builds. The [exact build commands](../.cumetal-artifacts/four-self-tests-4a207e2/build-commands.json)
and [replay runner](../.cumetal-artifacts/four-self-tests-4a207e2/run.py)
record all options, per-mode named selections, deadlines and environment.

GPU invocation: `BIN --cumetal-root PACKAGE --ptx self_test_MODE.ptx self-test
--check MODE.CHECK ...`, selecting every listed check belonging to the mode.
Set `CUMETAL_TRACE_GPU=1` and `CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0`.
Run the four CPU groups first, then the eight GPU attempts serially.

Local evidence: [metadata](../.cumetal-artifacts/four-self-tests-4a207e2/metadata.json),
[results and commands](../.cumetal-artifacts/four-self-tests-4a207e2/results.json),
[summary](../.cumetal-artifacts/four-self-tests-4a207e2/summary.json), and
[per-run logs](../.cumetal-artifacts/four-self-tests-4a207e2/runs/). These files
are local retained evidence, not public GitHub attachments. No ticket was closed.
