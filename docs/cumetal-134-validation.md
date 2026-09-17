# CuMetal #134 consumer validation

## Result at `f7ceeef`

[Issue #134](https://github.com/Lulzx/cuda-metal/issues/134), implemented by
[PR #139](https://github.com/Lulzx/cuda-metal/pull/139) at
`f7ceeeff731181112e691f3b3dbc872c17ef7743`, clears the original LLVM21 Bitcoin
`%r1688` packed-value narrowing failure. The normal immutable consumer replay
then stops at **IR verification**, before Metal compilation or GPU execution.
[Bitcoin self-test issue #30](https://github.com/brandonros/vanity-miner-rs/issues/30)
remains open. Earlier workload results remain in the
[#130 validation report](cumetal-130-validation.md).

| Check | Recorded outcome |
| --- | --- |
| Matching CPU groups | **147 pass:** Shallenge 21, Ethereum 8, Bitcoin 39, Solana 79 |
| Unchanged LLVM21 Bitcoin PTX through normal CuMetal CLI | Exit **1** after **94.578 s**, within the 600-second deadline |
| First compiler error | PTX line **32188**: `operand type ptr<device, i8> does not match value %197406 type i64` |
| Bitcoin check inventory | All **39** selected names reported once, blocked by compilation |
| Full Bitcoin GPU launches | **0** |

The CLI log contains **one slice of 144 distinct pointer/integer diagnostics**,
each appearing once, followed by the shared compilation failure reported for
each of the 39 check names. Its `0 passed, 39 failed` summary represents blocked
checks, not 39 numerical mismatches or 39 independent compiler attempts. The
earlier `%r1688` diagnostic is absent. This elapsed time is a bounded acceptance
observation, not an isolated performance benchmark.

## Implemented scope and local tests

The normalizer replaces a single `mov.b64` pack consumed only by exact
`cvt.u16.u64` or `cvt.u32.u64` with a conversion from its stable low32 source.
It preserves the destination format and wider register storage extension.
Declarations establish widths; whole-function definition/use accounting,
same-block unpredicated instructions, and intervening-write/call checks bound
the proof. Budget exhaustion leaves the original function unchanged. It does
not supply values for undefined high bits or implement general partial-value
analysis.

The committed [upstream validation report](https://github.com/brandonros/cuda-metal/blob/f7ceeeff731181112e691f3b3dbc872c17ef7743/docs/ptx-narrow-pack-validation.md)
records Apple M5 / macOS 26.6.2, LLVM/Clang 21.1.8, and successful builds:

| Configuration | Unit suite | PTX functional suite |
| --- | --- | --- |
| Release, binary shim OFF | 56 pass, 1 benchmark-precondition skip | 57 pass, 1 offline-Metal-tool skip |
| Debug, binary shim ON | 59 pass, same skip | 57 pass, same skip |

Both unit runs exclude `unit_metallib_parser` and `unit_cumetal_cli` because their
Apple reference/CLI prerequisites are unavailable. These are scoped suite
results, not an all-project pass.

The focused GPU runner passes **12 fixtures × 65 lanes × 5 output words** in
each configuration: unsigned 16/32-bit conversion, natural/wider64 storage, and
straight/intervening-work/loop layouts. A fully defined reference pack also has
an observable full-width use. Checks cover exact CPU results, full storage,
side effects, output guards, input immutability, ABI and Apple GPU provenance.
These are 780 lane positions per configuration, not 780 distinct inputs.

All **44 rejection controls** remain rejected. They include observable undefined
bits and deliberately unsupported proof shapes such as source overwrites,
intervening calls and multiple uses; those categories are distinct. The parent
compiler rejects all 12 positive fixtures and all 44 controls. Host tests also
cover declaration ambiguity, call returns, vector/tuple writes, compact ranges,
proof budgets and transactional exhaustion. These focused GPU results do not
establish full Bitcoin execution.

## Next blocker: mixed vector records

The first failing value is `%rd78`, loaded with scalar length `%rd79` by
`ld.local.v2.b64` at PTX line 32160. Its producer stores `{1,0}` for an empty
suffix. The zero-length guard bypasses byte reads; **literal 1 is not a proven
device pointer**. Correct handling still needs a guarded-use/reaching-write
proof or another justified representation of the empty slice. Inventing pointer
provenance would conceal the problem.

The neighboring reload at line 32159 contains a real initialized-global pointer
to `bc1q` and length 4. Its later errors directly confirm
[#118's mixed pointer/length vector-reload gap](https://github.com/Lulzx/cuda-metal/issues/118#issuecomment-5706131679).
The original pointer's address space must survive per-lane typing. The 144
diagnostics comprise 82 in the matching helper, 50 in the prefix/suffix probe,
and 12 in the length-boundary probe. The later probe groups were located and
counted, not independently reduced.

#118 is the next owner for the confirmed pointer-lane defect, with the empty
suffix retained as an additional proof boundary. Fixing proven pointer lanes
alone is not claimed to clear the sentinel group or the full module. No full
Bitcoin Metal source, GPU numerical result or workload acceptance is established.

## Reproduction identity and retained evidence

The consumer uses isolated miner source
`80699f24bd477b5551bed1d8eff3bc910a833210`, with only the CuMetal flake input/lock
updated. Isolation avoids the shared checkout's concurrent Rust-CUDA manifest
migration. The recorded source compatibility diff is empty. The CPU binary was
reused from `four-self-tests-4a207e2` with unchanged host logic and the identity
below; all four CPU groups were rerun.

| Artifact | Exact identity |
| --- | --- |
| CuMetal revision | `f7ceeeff731181112e691f3b3dbc872c17ef7743` |
| Nix package | `/nix/store/1i22rrss68v810vh6nrqh1rag9r4z15y-vanity-cumetal-f7ceeeff7311` |
| Nix source | `/nix/store/q4p0m95zlymp9lha7hwjisrszi7qwpsp-source` |
| Source NAR hash | `sha256-ZCTAsGjXoHLQM+beH4c+S+2tVgtnLJUzP/8TKhtEnO4=` |
| Compiler SHA-256 (`bin/cumetalc` under package) | `ac7f979208bef64d417471495e27e47b88e98a588a904dfc9d41748e0e3a4e90` |
| Runtime SHA-256 (`lib/libcumetal.dylib` under package) | `cd26161257cd0ae952b04f5edf16ea9deb7547e50f498bbf359cf2145de9e71f` |
| Normal CuMetal CLI SHA-256 | `f4941b36d133b67cbe2a94b294b83cd4efff50a076036339b5c9ff09e322cac8` |
| CPU CLI SHA-256 | `cbeb9593afaf1c257915c6ca7f83b4326b045b366f20269b46031aed9dd528f7` |
| Replay runner SHA-256 | `712fdf745fa6ee9612bc06f652e87a95ba63442a52261b90923f31d5d16cc8c1` |
| PTX producer | `afe80210ea28748cc58c3ba75f877bfa4a6b1ecd` |
| LLVM21 `self_test_bitcoin.ptx` SHA-256 | `48a78769d28dc0e699b2e3209b332e8cd6efa5d3846cf39d1b9b0776adbe035b` |

The 6,359,000-byte PTX comes from
[Actions run 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652),
entry `kernel_self_test_bitcoin`. The normal CLI receives the original PTX and
matched Nix package through `--ptx` and `--cumetal-root`; recorded identity
checks pass. `CUMETAL_TRACE_GPU=1` and
`CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0` are set.

Local evidence is retained under `.cumetal-artifacts/issue-134-f7ceeef/`:
`metadata.json`, `results.json`, original `inputs/llvm21/self_test_bitcoin.ptx`,
and `runs/llvm21-bitcoin/cli.log` (SHA-256
`c40666c9546647dd040a75f9be0f2e0e19befa4e804684c024254f324ebeb54b`).
The manifests retain the exact CLI command and all 39 selected names. Scoped
local suite commands/results and source-chain analysis are retained separately
under `upstream-issue-breakdown/issue-134-narrow-pack/` as `test-results.json`
and `next-blocker.md`. The earlier local-compiler 97.315-second translation
experiment in the upstream report is separate from this 94.578-second immutable
consumer replay.

Published and read back on [PR #139](https://github.com/Lulzx/cuda-metal/pull/139#issuecomment-5706130104),
[upstream #134](https://github.com/Lulzx/cuda-metal/issues/134),
[#118](https://github.com/Lulzx/cuda-metal/issues/118#issuecomment-5706131679),
and [downstream #30](https://github.com/brandonros/vanity-miner-rs/issues/30).

## Publication source build

The report/pin publication is based on miner
`8d59d1f90deda1e16d3029952177d528cbe46e4b`, which includes the separately committed
Rust-CUDA dependency migration. A clean, isolated normal CLI release build with
`cumetal,p256-public-key,self_test_solana,self_test_bitcoin,self_test_ethereum,self_test_shallenge`
passes under `nix develop .#cumetal` and `cargo build --locked` in 19.40 seconds.
Its CLI SHA-256 is `f4941b36d133b67cbe2a94b294b83cd4efff50a076036339b5c9ff09e322cac8`.
This is build validation only; the GPU measurements above still use isolated
`80699f24`, and no new PTX was generated by the publication build.
