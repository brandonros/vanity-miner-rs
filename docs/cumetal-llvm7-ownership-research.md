# CuMetal LLVM 7 ownership research: downstream #38 and #39

Compiler revision: `9e3e61574b776424a96c686bdbdc04ad1f27fe9f`; compiler SHA-256 `5662a763e6e63539cd9776efb844187c707d23297c35d2950c1e5820eb9ea628`.
No compiler implementation changes or GPU runs. Full-workload failures are the retained split-bundle observations; the new executions below are small probes and diagnostic input controls.

## Shallenge (#38)

The first failure is integer-minus-pointer inside a proven same-base cancellation. Existing #45/#56 explicitly exclude it; #76 conversion/join typing is not the matching mechanism. The five-instruction address expression reduces to `31 - length - prologue_bytes`. A diagnostic full-input scalar rewrite emits MSL. Published as [upstream #129](https://github.com/Lulzx/cuda-metal/issues/129); findings attached to [downstream #38](https://github.com/brandonros/vanity-miner-rs/issues/38#issuecomment-5691622619).

## P-256 public-key production (#39)

Scalar zero-marker propagation is missing before payload definedness analysis. Both signed and unsigned zero tests fail; a direct-predicate control emits MSL. A first full-input redundant guard clears `%rd18761`, and a second clears `%rd18765` before reaching `%rd18781`. The full workload remains blocked. This compact case is distinct from #83 empty-label/partial-rewrite and #120 depth cases; published as [upstream #130](https://github.com/Lulzx/cuda-metal/issues/130), with findings attached to [downstream #39](https://github.com/brandonros/vanity-miner-rs/issues/39#issuecomment-5691622840).

## Executed probes

| Input | Exit | Seconds | First output |
|---|---:|---:|---|
| `address-cancel` | 1 | 0.041 | cumetalc failed: line 13: pointer subtraction requires a pointer minus a 64-bit integer byte offset |
| `address-cancel-scalar-control` | 0 | 0.021 | MSL emitted |
| `sentinel-signed` | 1 | 0.021 | cumetalc failed: PTX register '%rd4' is undefined on an incoming edge to block 'JOIN' |
| `sentinel-unsigned` | 1 | 0.021 | cumetalc failed: PTX register '%rd4' is undefined on an incoming edge to block 'JOIN' |
| `sentinel-predicate-control` | 0 | 0.019 | MSL emitted |
| `sentinel-invalid-observable` | 1 | 0.019 | cumetalc failed: PTX register '%rd4' is undefined on an incoming edge to block 'JOIN' |
| `p256-guard-control` | 1 | 10.051 | cumetalc failed: PTX register '%rd18765' is undefined on an incoming edge to block '$L__BB5_1' |
| `shallenge-scalar-control` | 0 | 0.399 | MSL emitted |
| `sentinel-dynamic-marker` | 1 | 0.071 | cumetalc failed: PTX register '%rd4' is undefined on an incoming edge to block 'JOIN' |
| `sentinel-overwritten-invalid` | 1 | 0.020 | cumetalc failed: PTX register '%rd4' is undefined on an incoming edge to block 'JOIN' |
| `p256-two-guards-control` | 1 | 8.944 | cumetalc failed: PTX register '%rd18781' is undefined on an incoming edge to block '$L__BB5_1' |

NVIDIA CUDA 12.9 ptxas accepts `address-cancel`, `sentinel-signed`, `sentinel-unsigned`, and `sentinel-dynamic-marker` (exit 0 for each). The modular Shallenge identity passes 155 independent Python combinations. Neither result establishes GPU correctness.

Full public reproducers, evidence, proposed acceptance, source links and hashes are in the paired upstream issue bodies retained under `.cumetal-artifacts/research-38-39/`. Both upstream issue bodies and downstream research comments were read back and verified after publication.

## Recorded identities

- Compiler: `/nix/store/hh6l39zigklrh3al2w21x7b8ik5jaxcm-vanity-cumetal-9e3e61574b77/bin/cumetalc`; SHA-256 `5662a763e6e63539cd9776efb844187c707d23297c35d2950c1e5820eb9ea628`.
- Paired runtime (unused for execution): `/nix/store/hh6l39zigklrh3al2w21x7b8ik5jaxcm-vanity-cumetal-9e3e61574b77/lib/libcumetal.dylib`; SHA-256 `2b270e4154b9c16df64ee75f33655a37d7fea843a61055ff180ce3bd515ba549`.
- Full PTX producer: `7484a5dc76a0f1b4beed5ff0b75f248a91276d8e` plus recorded timing instrumentation.
- Original Shallenge PTX SHA-256: `bfbd79fec26057bfc77293777f3acbcb40cb84c61041274e2c66fe6770639e51`.
- Original P-256 public-key PTX SHA-256: `dbd5ada9e96613eb7f2da98e97b95310a5a6368744c5f8d1d84b9c184184e0da`.
- Original inputs: `.cumetal-artifacts/ptx-compile-ftsphb4j/llvm7/{shallenge,p256_public_key}.ptx`; diagnostic copies have separate filenames/hashes in `research-38-39/probes.json`.
