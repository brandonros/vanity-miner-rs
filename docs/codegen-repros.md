# Focused Rust reproductions of observed compiler failures

Tracking: [#19](https://github.com/brandonros/vanity-miner-rs/issues/19). Base inventory: 118 numerical slots plus the launch probe at `6a5b9e6`. These remain unchanged.

## Audit

The 36 tracked CuMetal fixes are not 36 missing Rust tests. The table distinguishes relevant existing source coverage from a verified reproduction of a specific emitted PTX shape. A slot named here is **not** a claim that reverting that individual compiler fix has been tested against that slot. Historic full-module failures and the upstream issue reports remain the evidence for attribution.

| CuMetal issue / PR | Existing Rust coverage or observed source | Decision |
| --- | --- | --- |
| [#23](https://github.com/Lulzx/cuda-metal/issues/23) / [#49](https://github.com/Lulzx/cuda-metal/pull/49) | All kernel launches | Host launch ABI; retain upstream test. |
| [#26](https://github.com/Lulzx/cuda-metal/issues/26) / [#50](https://github.com/Lulzx/cuda-metal/pull/50) | Slots 3, 41–44, 60–63: byte outputs | Existing Rust behavior; exact store width is a PTX check. |
| [#32](https://github.com/Lulzx/cuda-metal/issues/32) / [#50](https://github.com/Lulzx/cuda-metal/pull/50) | Slots 60–63, 94: narrow loads | Existing small Rust cases; retain opcode-width checks upstream. |
| [#44](https://github.com/Lulzx/cuda-metal/issues/44) / [#50](https://github.com/Lulzx/cuda-metal/pull/50) | Slots 46–56, 65–70: width/carry/shift arithmetic | Existing arithmetic cases; upstream checks exact source widths. |
| [#47](https://github.com/Lulzx/cuda-metal/issues/47) / [#50](https://github.com/Lulzx/cuda-metal/pull/50) | Slots 94, 96, 106: narrow values and returns | Existing small Rust coverage; parameter-store truncation checked upstream. |
| [#48](https://github.com/Lulzx/cuda-metal/issues/48) / [#50](https://github.com/Lulzx/cuda-metal/pull/50) | Slots 46–56, 94: integer conversions | Existing behavior; PTX source/container-width distinction checked upstream. |
| [#25](https://github.com/Lulzx/cuda-metal/issues/25) / [#52](https://github.com/Lulzx/cuda-metal/pull/52) | Full multi-entry module | Selection/validation policy; retain upstream test. |
| [#30](https://github.com/Lulzx/cuda-metal/issues/30) / [#52](https://github.com/Lulzx/cuda-metal/pull/52) | Slots 60–63; larger curve tables | Existing table coverage; relocation syntax stays upstream. |
| [#31](https://github.com/Lulzx/cuda-metal/issues/31) / [#53](https://github.com/Lulzx/cuda-metal/pull/53) | Device calls in full module | Parser formatting; no new Rust computation. |
| [#27](https://github.com/Lulzx/cuda-metal/issues/27) / [#54](https://github.com/Lulzx/cuda-metal/pull/54) | Array/aggregate outputs throughout suite | Literal PTX tuple syntax; no forced Rust syntax case. |
| [#28](https://github.com/Lulzx/cuda-metal/issues/28) / [#54](https://github.com/Lulzx/cuda-metal/pull/54) | Slots 0, 1, 6–9, 44: RNG/hash rotations | Existing operation coverage; inspect PTX per toolchain. |
| [#29](https://github.com/Lulzx/cuda-metal/issues/29) / [#54](https://github.com/Lulzx/cuda-metal/pull/54) | Slots 1, 6–9: hash byte operations | Existing behavior; emitted prmt is toolchain-dependent. |
| [#33](https://github.com/Lulzx/cuda-metal/issues/33) / [#54](https://github.com/Lulzx/cuda-metal/pull/54) | Packed arithmetic/curve internals | No dedicated Rust bfi guarantee; do not force unrelated Rust features. |
| [#34](https://github.com/Lulzx/cuda-metal/issues/34) / [#54](https://github.com/Lulzx/cuda-metal/pull/54) | Byte packing in hash/curve primitives | PTX tuple form depends on lowering; retain focused PTX regression. |
| [#36](https://github.com/Lulzx/cuda-metal/issues/36) / [#55](https://github.com/Lulzx/cuda-metal/pull/55) | Base58/nonce length handling | PTX overload typing; existing integration plus upstream unit test. |
| [#38](https://github.com/Lulzx/cuda-metal/issues/38) / [#55](https://github.com/Lulzx/cuda-metal/pull/55) | Slot 39: u64 high multiply; 49, 51, 59, 64 | Already dedicated arithmetic coverage; no duplicate kernel. |
| [#37](https://github.com/Lulzx/cuda-metal/issues/37) / [#56](https://github.com/Lulzx/cuda-metal/pull/56) | Curve/local-pointer internals | Dead conversion elimination; a standalone Rust expression may disappear. |
| [#39](https://github.com/Lulzx/cuda-metal/issues/39) / [#56](https://github.com/Lulzx/cuda-metal/pull/56) | Slots 60–63, 99–103: indexing/slices/helper paths | Existing direct tests retained; helper attempt tracked separately in CuMetal #80. |
| [#45](https://github.com/Lulzx/cuda-metal/issues/45) / [#56](https://github.com/Lulzx/cuda-metal/pull/56) | Slots 61–63, 97–101, 108: slice/index arithmetic | Existing Rust cases plus runtime-pointer paths in new kernel. |
| [#35](https://github.com/Lulzx/cuda-metal/issues/35) / [#57](https://github.com/Lulzx/cuda-metal/pull/57) | Slots 41–43, 105, 107–108: guarded base58/slices | Existing Rust bisects; no general CFG feature expansion. |
| [#40](https://github.com/Lulzx/cuda-metal/issues/40) / [#58](https://github.com/Lulzx/cuda-metal/pull/58) | Slots 0, 44: RNG seeding/nonce generation | New runtime nonce extraction exercises actual library path; no invented recursion. |
| [#41](https://github.com/Lulzx/cuda-metal/issues/41) / [#58](https://github.com/Lulzx/cuda-metal/pull/58) | Slots 96, 99–101, 106, 110: aggregate/copy/return | Existing small Rust cases; exact PTX parameter lanes stay upstream. |
| [#42](https://github.com/Lulzx/cuda-metal/issues/42) / [#58](https://github.com/Lulzx/cuda-metal/pull/58) | Slots 44, 61–62, 99: local arrays and iteration | Existing logical coverage; runtime nonce output isolates actual path. |
| [#43](https://github.com/Lulzx/cuda-metal/issues/43) / [#62](https://github.com/Lulzx/cuda-metal/pull/62) | Bounds/assertion paths in generated module | Trap reporting is a runtime contract; do not add deliberate panics to successful miner tests. |
| [#46](https://github.com/Lulzx/cuda-metal/issues/46) / [#62](https://github.com/Lulzx/cuda-metal/pull/62) | Assertions in reachable helper calls | Trap propagation stays in upstream runtime/compiler tests. |
| [#24](https://github.com/Lulzx/cuda-metal/issues/24) / [#63](https://github.com/Lulzx/cuda-metal/pull/63) | All kernel launches | Runtime provenance; retain upstream test. |
| [#59](https://github.com/Lulzx/cuda-metal/issues/59) / [#68](https://github.com/Lulzx/cuda-metal/pull/68) | Production input/output buffers | New runtime two-buffer nonce kernel exercises provenance; store-specific mutation remains upstream. |
| [#64](https://github.com/Lulzx/cuda-metal/issues/64) / [#69](https://github.com/Lulzx/cuda-metal/pull/69) | Production Shallenge; fixed nonce slot 44 did not catch runtime mismatch | High-value gap: new nonce vectors span seeds, indices, and zero/multiple iterations. |
| [#51](https://github.com/Lulzx/cuda-metal/issues/51) / [#70](https://github.com/Lulzx/cuda-metal/pull/70) | Slots 84, 102–103, 109, 112–117: scalar reductions/zero paths | Already dedicated recovered Rust slots; do not introduce undefined Rust halves. |
| [#66](https://github.com/Lulzx/cuda-metal/issues/66) / [#70](https://github.com/Lulzx/cuda-metal/pull/70) | Same recovered scalar-zero slots as #51 | Keep existing Rust cases; named discarded PTX registers are allocator-dependent. |
| [#60](https://github.com/Lulzx/cuda-metal/issues/60) / [#71](https://github.com/Lulzx/cuda-metal/pull/71) | Production scalar lengths/seeds | New nonce input buffer avoids constant-only data; launch scalar classification remains upstream. |
| [#65](https://github.com/Lulzx/cuda-metal/issues/65) / [#71](https://github.com/Lulzx/cuda-metal/pull/71) | Every typed kernel launch | ABI sidecar policy; upstream tests, not extra Rust arithmetic. |
| [#61](https://github.com/Lulzx/cuda-metal/issues/61) / [#72](https://github.com/Lulzx/cuda-metal/pull/72) | Production Shallenge pointer/length handling | Runtime nonce extraction; commuted PTX register order is not guaranteed by Rust source. |
| [#78](https://github.com/Lulzx/cuda-metal/issues/78) / [#79](https://github.com/Lulzx/cuda-metal/pull/79) | Upstream alias-write mutation audit | Not an observed miner computation; no new mutable-global Rust feature. |
| [#67](https://github.com/Lulzx/cuda-metal/issues/67) / [#74](https://github.com/Lulzx/cuda-metal/pull/74) | Recovered scalar-wide slot 103; existing table slots 60–63 | Helper attempt still fails current compiler (#80); retain existing slots and defer the additional kernel. |
| [#73](https://github.com/Lulzx/cuda-metal/issues/73) / [#75](https://github.com/Lulzx/cuda-metal/pull/75) | Upstream ReLU fixture, not an observed miner Rust kernel | Do not add ReLU or require specific register reuse merely to match upstream bug. |

## Bounded additions

One Rust-generated entry returns raw values so the runner reports the exact input and mismatch. It does not renumber or enlarge the existing 118-slot known-answer protocol and uses the existing `self_test` feature.

`kernel_repro_nonce_sequence` calls the real `generate_base64_nonce`, removing Shallenge hashing and match/output coordination. Its 240 cases span six seeds (including nonzero high halves and wrap), five thread indices, and eight lengths including zero. The CPU oracle calls the same production logic and asserts the fixed two-byte result `pe` for the smallest historical mismatch. Existing known-answer checks are retained.

Generated PTX retains runtime loads, loop paths, 64-bit tuple packs, and RNG seeding. No handwritten PTX or inline assembly is substituted.

A table/device helper was also investigated, because the existing direct table slots do not isolate that interaction. A faithful pointer/index helper still fails current typed import, so it is **not added to the accepted suite**. Its small Rust source, generated PTX and diagnostics are preserved in [CuMetal #80](https://github.com/Lulzx/cuda-metal/issues/80). The investigation also showed that `#[inline(never)]` alone does not stop argument promotion from replacing a plain pointer-reading helper with a byte identity function; that optimized-away case was not counted as coverage.

## Build and run

Build the ordinary full-feature PTX using the existing Rust-CUDA workflow or the documented Linux `nix develop .#v21` command. The PTX entry inventory now also requires this additional kernel. The PR build supplies LLVM 7/21 artifacts without running the release job.

```sh
cargo run -p logic --example codegen_repro_vectors --features self_test --locked > vectors.json
python3 scripts/run-codegen-repros.py --ptx output-llvm21.ptx --vectors vectors.json \
  --library /path/to/libcumetal.dylib --cumetalc /path/to/cumetalc --out repro-results
```

For NVIDIA CUDA, omit `--cumetalc` and supply the CUDA driver library. Select a single kernel with `--entry`. This is a one-thread diagnostic suite; it is not a launch-boundary or throughput test. The runner snapshots PTX/vectors/tools, records hashes, checks both buffer guards and unchanged inputs, and retains exact mismatches.

## Validation status

Rust source commit `62580f7` builds the nonce-only suite in [CI run 34877613237](https://github.com/brandonros/vanity-miner-rs/actions/runs/34877613237). All four compile cells pass (LLVM 7/21 on x86_64/aarch64). Its LLVM 21 artifact contains all 124 expected entries: four mining kernels, the original 118 numerical checks and probe, and the new nonce entry. All 40 host logic tests and all 118 existing CPU self-tests pass; their source bodies and slot assignments are unchanged.

The LLVM 21 nonce entry is 187 PTX lines, versus 2,731 for the production Shallenge entry (entry bodies, excluding reachable helpers). On the archived CuMetal `1e3c01e`, seed 0 / thread 0 / length 2 produces `p9` instead of `pe`. Corrected CuMetal `ddf496c` passes all 240 cases on Apple M5, with unchanged inputs and intact output guards. Both runs use exactly the same final PTX, vectors and runtime. Generated old MSL uses 32-bit variables for packed loop state where the corrected output uses 64 bits, matching the #64 failure class. This is a combined historical/current compiler comparison, not an isolated single-commit revert.

| Input | SHA-256 |
| --- | --- |
| Final LLVM 21 PTX | `a0de7749b4b146bc6e41a7628a9be7c336ea55239a3c41f97e126eeee455d1cb` |
| 240 vectors | `c1153d23a17727e4799df2d095109b12bd3512be9b8b329c8b9b660598640a00` |
| Corrected compiler | `b7cf0485d3dc7a392e42b81c47a54ad0275f11fc0e5caf45cf34eb735b2a6103` |
| Archived compiler | `af873d479c822b598867fb5a7478b1acdd1891b5a93053e1a725671fd47abc91` |
| Shared runtime | `31492c9be52ea10c000490f7a9435ffe899d5eabf40b91ceb233eec40b110d2e` |

The helper investigation reproduced its address-space verification failure both with and without a bounds-check branch. The no-branch pointer/index source is preserved at `e18d297`; its LLVM 21 artifact has SHA-256 `4e08de091993f2475cd12ee8ecb13620ead3683ec96a3e2f52c2f231416b4b12`. That unsupported helper is excluded from the final suite and tracked in CuMetal #80.

No NVIDIA numerical execution or full 118-entry GPU rerun is claimed for this change. The CI matrix validates Rust-CUDA compilation/export; the Apple M5 runs validate this reproduction, and the local CPU run preserves the existing known-answer inventory.
