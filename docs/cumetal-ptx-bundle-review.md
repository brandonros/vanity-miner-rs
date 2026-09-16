# New PTX bundles: issue-matrix comparison

Input record: `7484a5dc76a0f1b4beed5ff0b75f248a91276d8e` plus the recorded timing instrumentation. CuMetal: `9e3e61574b776424a96c686bdbdc04ad1f27fe9f`.

This reviews the existing serial translation run; it does not rerun or interrupt that process. Both bundle hashes and all 32 extracted PTX hashes match `artifacts/ptx-bundle-hashes.json`.

**Scope: PTX → Metal source only.** No Apple source compilation, pipeline creation, GPU execution or CPU verification is established by a passing cell. The prior [full GPU report](cumetal-status.md) tested different inputs and remains scoped to those inputs.

## Sixteen logical modules, two LLVM versions

| Module | Downstream issue | LLVM 7 translation | LLVM 21 translation |
| --- | --- | --- | --- |
| `solana` | [#23](https://github.com/brandonros/vanity-miner-rs/issues/23) | Metal IR: `mul.hi` operand types | PTX 34070: operand type i32 does not match value %35707 type i64 |
| `bitcoin` | [#24](https://github.com/brandonros/vanity-miner-rs/issues/24) | **MSL emitted** | Pointer-offset type error |
| `ethereum` | [#25](https://github.com/brandonros/vanity-miner-rs/issues/25) | **MSL emitted** | Pointer-offset type error |
| `shallenge` | [#38](https://github.com/brandonros/vanity-miner-rs/issues/38) | Pointer-offset type error | **MSL emitted** |
| `p256_public_key` | [#39](https://github.com/brandonros/vanity-miner-rs/issues/39) | Undefined `%rd18761` at `$L__BB5_1` | **MSL emitted** |
| `p256_signature` | [#26](https://github.com/brandonros/vanity-miner-rs/issues/26) | Vector parameter transfer rejected | Undefined `%rd17653` at `$L__BB0_11` |
| `rsa_modulus` | [#27](https://github.com/brandonros/vanity-miner-rs/issues/27) | Undefined `%rd1824` at `$L__BB0_38` | **MSL emitted** (advance only) |
| `rsa_pss` | [#28](https://github.com/brandonros/vanity-miner-rs/issues/28) | Vector parameter transfer rejected | **MSL emitted** |
| `self_test_solana` | [#29](https://github.com/brandonros/vanity-miner-rs/issues/29) | Undefined `%rd20` at `$L__BB77_4` | Undefined `%rd42` at `$L__BB77_6` |
| `self_test_bitcoin` | [#30](https://github.com/brandonros/vanity-miner-rs/issues/30) | Undefined `%rd29462` at `$L__BB48_1` | Undefined `%r1688` at `$L__BB4_3` |
| `self_test_ethereum` | [#31](https://github.com/brandonros/vanity-miner-rs/issues/31) | Undefined `%rd29462` at `$L__BB15_1` | Pointer-offset type error |
| `self_test_shallenge` | [#37](https://github.com/brandonros/vanity-miner-rs/issues/37) | Pointer-join type error | Undefined `%rd16` at `$L__BB18_1` |
| `self_test_p256_public_key` | [#32](https://github.com/brandonros/vanity-miner-rs/issues/32) | Undefined `%rd18761` at `$L__BB18_1` | **MSL emitted** |
| `self_test_p256_signature` | [#33](https://github.com/brandonros/vanity-miner-rs/issues/33) | Pointer-join type error | Undefined `%p243` at `$L__BB8_1` |
| `self_test_rsa_modulus` | [#34](https://github.com/brandonros/vanity-miner-rs/issues/34) | Undefined `%rd676` at `$L__BB4_1` | Undefined `%rs907` at `$L__BB16_1` |
| `self_test_rsa_pss` | [#35](https://github.com/brandonros/vanity-miner-rs/issues/35) | Pointer-join type error | **MSL emitted** |

## Totals and entry selection

- LLVM 7: 2 emit MSL, 14 fail translation, 0 time out; 16 module attempts.
- LLVM 21: 6 emit MSL, 10 fail translation, 0 time out; 16 module attempts.

The script supplies no `--entry`; the pinned importer selects the first entry. All modules have one entry except RSA modulus, where this is `kernel_rsa_advance`. The other three RSA stages were not selected by this sweep. Complete coverage would select all 19 entries per LLVM version (38 total).

## What this means for #76

The compiler correction plan and closure gates are in [the #76 plan](cumetal-76-plan.md). A blocker is attributed to the exact input and stage; a diagnosis from one LLVM version cannot automatically be assigned to the other.

- **LLVM 21 Solana, Bitcoin and Ethereum** reproduce the first errors recorded
  under #76: the i32/i64 disagreement at 34070 and pointer-offset errors at 26556
  and 22069. Retain #76 as their shared production typing correction.
- **LLVM 7 Bitcoin and Ethereum** can proceed to Apple compilation and verified
  GPU acceptance. Their MSL emission does not close #24/#25 or demonstrate a
  compiler correction for the LLVM 21 inputs.
- **LLVM 7 Solana** is a concrete #76 lead: its failing `mul.hi.u64` consumes a
  value joining 64-bit shifts with `cvt.u64.u32`. Reduce and verify the inferred
  incoming types before declaring it the same defect.
- **Tracking follow-up:** LLVM 7 Shallenge and P-256 public-key production fail
  translation, although their LLVM 21 counterparts emit MSL. Published downstream
  issues [#38](https://github.com/brandonros/vanity-miner-rs/issues/38) and
  [#39](https://github.com/brandonros/vanity-miner-rs/issues/39) now track those two
  production failures with version-specific evidence and acceptance. Upstream
  owners remain unconfirmed. Comparison comments were published on all other
  14 workload trackers; [the matrix](cumetal-issue-matrix.md#measured-translation-times-by-llvm-version)
  links directly to those comments. Existing full-GPU evidence and open statuses
  are retained.

Fresh diagnostics in changed inputs remain observations until reduced or traced. In particular, an undefined register or pointer-branch error is not automatically the same defect as a historical error with similar wording. The downstream workload trackers remain open through full acceptance; this review does not automatically change their upstream owner or close them.

## Reproduction record

- Compiler SHA-256: `5662a763e6e63539cd9776efb844187c707d23297c35d2950c1e5820eb9ea628`.
- Runtime SHA-256 (paired package verified by harness, unused for GPU execution here): `2b270e4154b9c16df64ee75f33655a37d7fea843a61055ff180ce3bd515ba549`.
- LLVM 7 archive SHA-256: `6309513c425ca9294eb64444e8b985f30b3934b42b165ab248c11c829949cbd0`.
- LLVM 21 archive SHA-256: `10664da849c22ee551be03cdf431d3fbd8183151fde2258ce7d60f63ed146f85`.
- Run began: `2026-09-16T02:45:36Z`; per-module elapsed times are in its `timings.tsv`. These are retained-cache translation timings, not clean-build benchmarks.
- Local evidence: `.cumetal-artifacts/ptx-compile-ftsphb4j/` (`metadata.json`, `timings.tsv`, per-module PTX/MSL/logs).
- Readback audit: `.cumetal-artifacts/new-ptx-matrix-review.json` records every selected entry, diagnostic and verified hash. These are local ignored artifacts, not public attachments.

The actual command pattern is:

```sh
/nix/store/hh6l39zigklrh3al2w21x7b8ik5jaxcm-vanity-cumetal-9e3e61574b77/bin/cumetalc INPUT.ptx --backend=cumetal-ir --ptx-strict --overwrite --emit=msl -o OUTPUT.metal
```
