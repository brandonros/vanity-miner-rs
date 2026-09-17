# CuMetal #130 consumer validation

## Self-test result at `0f98856`

The [scalar-guard implementation, draft PR #138](https://github.com/Lulzx/cuda-metal/pull/138)
clears the tested zero-marker translation failures. It does **not** make all
linked workloads pass. All 147 matching CPU checks pass again. Across the four
listed GPU attempts: **78 checks pass, one fails numerically, and 68 are blocked**.
No downstream self-test issue can close.

| Workload | Translation | Apple preparation | GPU result | Total seconds |
| --- | --- | --- | --- | ---: |
| LLVM21 Solana (#29) | Pass | Pass | **78 pass; `candidate_match` gets 0, expected 1** | 405.087 |
| LLVM7 Ethereum (#31) | Pass | Library creation passes; pipeline creation times out | 8 blocked, no completed launch | 600.013 |
| LLVM7 Bitcoin (#30) | Local pointer-memory proof budget exhausted | Not reached | 39 blocked | 97.530 |
| LLVM21 Shallenge (#37) | Pass | Source compilation rejects pointer qualifiers/casts | 21 blocked | 16.890 |

Solana has actual `generic_ptx_lowering`, `semantic_quality=exact`, Apple M5 GPU
provenance, one block/one thread. Its command-buffer interval is **32.333 ms**;
most elapsed time is preparation. `candidate_miss` and `candidate_invalid` pass,
so the failed matching case must not be summarized as full candidate acceptance.
The reduced private-record helper defect is tracked in [#140](https://github.com/Lulzx/cuda-metal/issues/140). A small private-record helper
control emits wrong pointer address spaces before and after #130; that is
separate evidence, not a measured earlier full-Solana GPU failure.

Ethereum samples near 120 and 512 seconds locate the wait in
`newComputePipelineStateWithFunction`, before dispatch. These serial invocations
use 600-second whole-process deadlines; times are not isolated API measurements
or cold-cache performance benchmarks. Only the owned client process group is
stopped on timeout. Solana completes within its deadline.

## Next blockers and ownership

- **Solana:** numerical owner [#140](https://github.com/Lulzx/cuda-metal/issues/140); private prefix/suffix pointers
  reloaded through a helper as device pointers are demonstrated by the published
  numerical reduction. Do not close
  #29 from its other 78 passing checks.
- **Ethereum:** [the LLVM7 evidence is published on #133](https://github.com/Lulzx/cuda-metal/issues/133#issuecomment-5706131380), which
  also retains a historical LLVM21 timeout. A shared compiler hotspot has
  not been established.
- **Bitcoin:** #76 records the new retained-facts resource limit separately from
  its earlier corrected alignment/range cases; exhaustion is not an overlap proof.
- **Shallenge:** [confirmed #118 source chains](https://github.com/Lulzx/cuda-metal/issues/118#issuecomment-5705687543)
  identify heterogeneous private pointer/length vector reloads. Correct helper
  specialization and complete execution remain unverified.

## Reproduction identity

The miner is built from isolated source `80699f24bd477b5551bed1d8eff3bc910a833210`
with only the CuMetal flake selection changed. This avoids a concurrent
Rust-CUDA manifest migration in the shared checkout. CLI, logic and kernel
sources match PTX producer `afe80210ea28748cc58c3ba75f877bfa4a6b1ecd`,
[Actions run 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
All original PTX hashes are checked before and after use.

- CuMetal: `0f98856b9bcab06f0c41d239a86684fdf0d6371d`.
- Package: `/nix/store/kjsn81k4167n0mdnkqkhvja5zimhskah-vanity-cumetal-0f98856b9bca`.
- Compiler SHA-256: `b933738d262d0b86edd175bc1f870ad828ae093b85e4f01f809e94562dc25f2f`.
- Runtime SHA-256: `4ed3c464be659c5e62a2fef2552bd3d8b348c1f5eb387f2937a505f366926159`.
- Normal CLI SHA-256: `05c35b7c8faf2557240269bf6c5e1c6a5150a835d4a0aa8b7382b8620ecb5e67`.

The normal CLI uses PTX and the matched immutable Nix package. Per-command
identity checks, CPU inventories, complete named outcomes where available,
GPU provenance, output guards and unchanged input checks are retained in
`.cumetal-artifacts/issue-130-0f98856/`: `metadata.json`, `results.json`,
`summary.json`, `retained-outputs.json` and per-run logs/samples.
The 68 blocked checks are not numerical failures. Ethereum's timeout yields
no per-check outcomes; its eight selected checks are explicitly counted blocked.

## P-256 production accepted on both LLVM versions

**[#39 is closed as completed](https://github.com/brandonros/vanity-miner-rs/issues/39).** The normal miner at immutable CuMetal `0f98856b9bcab06f0c41d239a86684fdf0d6371d` ([PR #138](https://github.com/Lulzx/cuda-metal/pull/138), implementing #130) passes all 11 bounded P-256 public-key invocations: **22 GPU batches, 704 CPU-reference-checked candidate positions, 18 selected winners**. The cumulative upstream PR remains unmerged; workload acceptance and upstream integration are separate statuses.

The original LLVM7 issue input (`dbd5ada9e96613eb7f2da98e97b95310a5a6368744c5f8d1d84b9c184184e0da`) passes an all-match run. Current Actions inputs from producer `afe80210` pass all five profiles under each LLVM version:

| Input | Profile | Total seconds | Selected winners |
| --- | --- | ---: | ---: |
| original | xy-all | 163.676 | 2 |
| llvm7 | xy-all | 165.640 | 2 |
| llvm7 | x-prefix | 11.797 | 2 |
| llvm7 | y-suffix | 12.076 | 2 |
| llvm7 | uncompressed | 11.821 | 2 |
| llvm7 | xy-no-match | 11.930 | 0 |
| llvm21 | xy-all | 151.842 | 2 |
| llvm21 | x-prefix | 7.401 | 2 |
| llvm21 | y-suffix | 7.396 | 2 |
| llvm21 | uncompressed | 7.591 | 2 |
| llvm21 | xy-no-match | 7.419 | 0 |

Each invocation uses two batches of 32 candidates with `--verify`, exact input/compiler/runtime identity checks and actual Apple M5 GPU provenance. Profiles cover all four output targets (`x`, `y`, `xy`, `uncompressed`), selective nonempty prefix/suffix checks, all-match controls and an all-zero XY pattern that yields no matches. Inputs and buffer guards pass. CPU reference checks aggregate match/error counts and the selected winner payload; host RustCrypto verification independently reconstructs each selected full public point and checks the pattern. This does **not** compare every candidate's complete point payload or establish exhaustive input coverage.

Candidates use the mode's `OsRng` seed; `--seed` is not claimed to control them. Counts are candidate positions, not unique-input counts. Every run finishes within 600 seconds. First invocations include long pipeline preparation; these are bounded acceptance timings, not cold-cache performance benchmarks.

Host source `80699f24` is isolated from the shared checkout's concurrent Rust-CUDA dependency migration. CLI/logic/kernel source matches producer `afe80210`; only the CuMetal input/lock changes. Original retained PTX is unchanged. No supplied MSL or alternate runtime bypasses the dependency gate.

- Compiler SHA-256: `b933738d262d0b86edd175bc1f870ad828ae093b85e4f01f809e94562dc25f2f`.
- Runtime SHA-256: `4ed3c464be659c5e62a2fef2552bd3d8b348c1f5eb387f2937a505f366926159`.
- CLI SHA-256: `05c35b7c8faf2557240269bf6c5e1c6a5150a835d4a0aa8b7382b8620ecb5e67`.
- Current llvm7 PTX SHA-256: `1bfb1151e81954c79e635dd20d189c4c52382c0ae5ffbdc242e6a3e6bbad8273`.
- Current llvm21 PTX SHA-256: `f0984f49610efe141e723ac313ef9670b5cbc9a118e7db2409383b1fbced2989`.

Reproduce with the recorded source/input and the matched Nix package:

```sh
vanity-miner --ptx /absolute/path/p256_public_key.ptx \
  --blocks 1 --threads-per-block 32 --batches 2 --verify \
  p256-public-key-vanity --target xy
```

Then use `--target x --prefix 0`, `--target y --suffix f`,
`--target uncompressed --prefix 04`, and `--target xy --prefix` followed by 128 zero hex digits. Keep each profile bounded to two batches. The latest compiler self-test results remain separate: #29/#30/#31/#37 stay open, including Solana's numerical issue #140.
