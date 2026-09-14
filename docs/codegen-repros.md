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
| [#39](https://github.com/Lulzx/cuda-metal/issues/39) / [#56](https://github.com/Lulzx/cuda-metal/pull/56) | Slots 60–63, 99–103: indexing/slices/helper paths | Probe shared table/device helper in the new focused case. |
| [#45](https://github.com/Lulzx/cuda-metal/issues/45) / [#56](https://github.com/Lulzx/cuda-metal/pull/56) | Slots 61–63, 97–101, 108: slice/index arithmetic | Existing Rust cases plus runtime-pointer paths in new kernels. |
| [#35](https://github.com/Lulzx/cuda-metal/issues/35) / [#57](https://github.com/Lulzx/cuda-metal/pull/57) | Slots 41–43, 105, 107–108: guarded base58/slices | Existing Rust bisects; no general CFG feature expansion. |
| [#40](https://github.com/Lulzx/cuda-metal/issues/40) / [#58](https://github.com/Lulzx/cuda-metal/pull/58) | Slots 0, 44: RNG seeding/nonce generation | New runtime nonce extraction exercises actual library path; no invented recursion. |
| [#41](https://github.com/Lulzx/cuda-metal/issues/41) / [#58](https://github.com/Lulzx/cuda-metal/pull/58) | Slots 96, 99–101, 106, 110: aggregate/copy/return | Existing small Rust cases; exact PTX parameter lanes stay upstream. |
| [#42](https://github.com/Lulzx/cuda-metal/issues/42) / [#58](https://github.com/Lulzx/cuda-metal/pull/58) | Slots 44, 61–62, 99: local arrays and iteration | Existing logical coverage; runtime nonce output isolates actual path. |
| [#43](https://github.com/Lulzx/cuda-metal/issues/43) / [#62](https://github.com/Lulzx/cuda-metal/pull/62) | Bounds/assertion paths in generated module | Trap reporting is a runtime contract; do not add deliberate panics to successful miner tests. |
| [#46](https://github.com/Lulzx/cuda-metal/issues/46) / [#62](https://github.com/Lulzx/cuda-metal/pull/62) | Assertions in reachable helper calls | Trap propagation stays in upstream runtime/compiler tests. |
| [#24](https://github.com/Lulzx/cuda-metal/issues/24) / [#63](https://github.com/Lulzx/cuda-metal/pull/63) | All kernel launches | Runtime provenance; retain upstream test. |
| [#59](https://github.com/Lulzx/cuda-metal/issues/59) / [#68](https://github.com/Lulzx/cuda-metal/pull/68) | Production input/output buffers | New runtime two-buffer kernels exercise provenance; store-specific mutation remains upstream. |
| [#64](https://github.com/Lulzx/cuda-metal/issues/64) / [#69](https://github.com/Lulzx/cuda-metal/pull/69) | Production Shallenge; fixed nonce slot 44 did not catch runtime mismatch | High-value gap: new nonce vectors span seeds, indices, and zero/multiple iterations. |
| [#51](https://github.com/Lulzx/cuda-metal/issues/51) / [#70](https://github.com/Lulzx/cuda-metal/pull/70) | Slots 84, 102–103, 109, 112–117: scalar reductions/zero paths | Already dedicated recovered Rust slots; do not introduce undefined Rust halves. |
| [#66](https://github.com/Lulzx/cuda-metal/issues/66) / [#70](https://github.com/Lulzx/cuda-metal/pull/70) | Same recovered scalar-zero slots as #51 | Keep existing Rust cases; named discarded PTX registers are allocator-dependent. |
| [#60](https://github.com/Lulzx/cuda-metal/issues/60) / [#71](https://github.com/Lulzx/cuda-metal/pull/71) | Production scalar lengths/seeds | New nonce input buffer avoids constant-only data; launch scalar classification remains upstream. |
| [#65](https://github.com/Lulzx/cuda-metal/issues/65) / [#71](https://github.com/Lulzx/cuda-metal/pull/71) | Every typed kernel launch | ABI sidecar policy; upstream tests, not extra Rust arithmetic. |
| [#61](https://github.com/Lulzx/cuda-metal/issues/61) / [#72](https://github.com/Lulzx/cuda-metal/pull/72) | Production Shallenge pointer/length handling | Runtime nonce extraction; commuted PTX register order is not guaranteed by Rust source. |
| [#78](https://github.com/Lulzx/cuda-metal/issues/78) / [#79](https://github.com/Lulzx/cuda-metal/pull/79) | Upstream alias-write mutation audit | Not an observed miner computation; no new mutable-global Rust feature. |
| [#67](https://github.com/Lulzx/cuda-metal/issues/67) / [#74](https://github.com/Lulzx/cuda-metal/pull/74) | Recovered scalar-wide slot 103; existing table slots 60–63 | Add shared-helper table/device path; verify emitted address conversions before claiming exact repro. |
| [#73](https://github.com/Lulzx/cuda-metal/issues/73) / [#75](https://github.com/Lulzx/cuda-metal/pull/75) | Upstream ReLU fixture, not an observed miner Rust kernel | Do not add ReLU or require specific register reuse merely to match upstream bug. |

## Bounded additions

Two separate Rust-generated entries return raw values so the runner reports the exact input and mismatch. They do not renumber or enlarge the existing 118-slot known-answer protocol. Both use the existing `self_test` feature.

- `kernel_repro_nonce_sequence`: calls the real `generate_base64_nonce`, removing Shallenge hashing and match/output coordination. 240 cases span six seeds (including nonzero high halves and wrap), five thread indices, and eight lengths including zero. The CPU oracle calls the same production logic; existing fixed known-answer tests remain its independent checks.
- `kernel_repro_alphabet_helper`: the production nonce alphabet and a runtime device buffer pass through the same non-inlined byte-reading helper. 129 cases check all alphabet indices, masked large indices, and input byte values against an independent literal alphabet. This extends the existing direct table slots with the helper/storage interaction.

The generated PTX must be inspected before treating either as an exact reproduction of #64 or #67. A passing related path is useful coverage but does not prove that the historical failing instruction shape survived reduction. No handwritten PTX or inline assembly is substituted.

## Build and run

Build the ordinary full-feature PTX using the existing Rust-CUDA workflow or the documented Linux `nix develop .#v21` command. The PTX entry inventory now also requires these two kernels. The PR build supplies LLVM 7/21 artifacts without running the release job.

```sh
cargo run -p logic --example codegen_repro_vectors --features self_test --locked > vectors.json
python3 scripts/run-codegen-repros.py --ptx output-llvm21.ptx --vectors vectors.json \
  --library /path/to/libcumetal.dylib --cumetalc /path/to/cumetalc --out repro-results
```

For NVIDIA CUDA, omit `--cumetalc` and supply the CUDA driver library. Select a single kernel with `--entry`. This is a one-thread diagnostic suite; it is not a launch-boundary or throughput test. The runner snapshots PTX/vectors/tools, records hashes, checks both buffer guards and unchanged inputs, and retains exact mismatches. No-match CLI semantics are outside this issue.

## Validation status

Host vector generation and logic tests are available locally. Rust-CUDA CI compilation, emitted-code inspection, current CuMetal execution, and historical comparison are pending; none is inferred from CPU success.
