# Shallenge production validation on CuMetal `4a207e2`

Measured **2026-09-16**, using the normal CLI rebuilt against the locked, matched
Nix compiler/runtime package. **Both LLVM7 and LLVM21 pass.** This validates
production `kernel_shallenge`; the separate Shallenge self-tests were not rerun.

The fix is [cuda-metal PR #132](https://github.com/Lulzx/cuda-metal/pull/132),
implementing [#129](https://github.com/Lulzx/cuda-metal/issues/129), at exact head
`4a207e277249370573d7fe170bb7b3346d9da015`. It depends on draft #131 at `0242f22`.
**Downstream [#38](https://github.com/brandonros/vanity-miner-rs/issues/38) is closed
as completed. Upstream #129 remains open for PR integration; its measured
compiler and production acceptance gates pass.** Shallenge self-tests remain
tracked separately in [#37](https://github.com/brandonros/vanity-miner-rs/issues/37).

## Results

| PTX | Profiles | GPU batches | CPU-reference-checked positions | Independent winner SHA-256 checks | First CLI run | Subsequent runs |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LLVM7 | **33/33 pass** | 66 | 2,112 | 48 | 3.443 s | 0.834–0.896 s |
| LLVM21 | **33/33 pass** | 66 | 2,112 | 48 | 1.259 s | 0.831–0.927 s |
| **Total** | **66/66 pass** | **132** | **4,224** | **96** | | |

Each version covers every valid username byte length (1–30), username `miner`
with seeds 1 and 2, and a zero-target no-match case. Every invocation runs two
batches of 32 candidates with `--verify`, checks input immutability and buffer
guards, and records exact generic PTX lowering on the Apple M5 GPU with the
expected launch dimensions. Both no-match profiles return no winners. The
maximum initial target tightens after a winner; later candidates need not match.

The CPU recomputes all candidate positions and compares aggregate match/error
counts plus each selected winner's full payload. An independent Python SHA-256
check verifies all 96 published 32-byte challenges, nonce alphabets and decreasing
hashes. Every LLVM7/LLVM21 profile pair returns identical winner records. The
kernel returns one winner per batch, so this does not compare every GPU
candidate payload or claim 4,224 unique inputs.

No run timed out under the 600-second per-invocation deadline. These are
correctness runs, not controlled-cache throughput or compilation benchmarks.
An independent read-only audit rehashed all artifacts and logs and recomputed
coverage, launch counts, totals and winner hashes without discrepancies.

## What changed

LLVM7 forms `base + (1 - cursor - length) + 30`, with
`cursor = base + written`. CuMetal now proves that the shared base cancels and
emits the wrapping scalar count `31 - length - written`. The pass tracks offsets
through reaching SSA definitions, branches and loops, then rebuilds type and
memory proofs from the rewritten instructions. The original PTX is unchanged.

Unrelated bases, observable intermediate addresses, mixed address representations
and unproven scalar-to-address uses remain rejected. Regression tests include
spilling/reloading the recovered count and passing it to a pointer-taking helper,
so obsolete pointer facts cannot make a scalar address appear valid.

Release (binary shim off) passes 45 selected unit tests; Debug (shim on) passes
48. Both pass 50 PTX functional tests and six typed-CUDA functional tests, with
one offline-Metal-tool skip and one unsupported wide-atomics skip per configuration.
`unit_cumetal_cli` is excluded; this is not a full repository-suite pass. The
new target has 84 arithmetic checks, five structural acceptance cases and 15
negatives. Eight GPU variants check 520 outputs per configuration, including
wrapping boundaries, negation, selection, loops, branches and scalar controls.
The final helper pair was rerun in both configurations after the selected suites.

Against immutable baseline `0242f22`, five cancellation variants fail translation
and unary negation fails Apple compilation; both scalar controls pass. The new
compiler passes all eight. This makes the regression coverage distinguish the
fix from the earlier implementation.

## Identity and reproduction

- Miner host source base: `2cd263fb1a28b784dd3967302ead48f5c72f6452`.
- PTX producer: `afe80210ea28748cc58c3ba75f877bfa4a6b1ecd`.
- CuMetal revision: `4a207e277249370573d7fe170bb7b3346d9da015`.
- Matched package: `/nix/store/is7s91d1vjk6ja2i3ldb4ijks3ccfygc-vanity-cumetal-4a207e277249`.
- Compiler SHA-256: `365e522b8c184233ec2fd1324cfee9b397adbfad121d90e1a99ab88e714c49a4`.
- Runtime SHA-256: `b3f1b7cd5ae3517cab97632b6b73a2a664930b6930334b7c5be02ab31c4b510d`.
- Host CLI SHA-256: `3060c5c49e0747385c13feae5b3fe18ceb9be9e4698c51411ead3ffbe5de5307`.
- LLVM7 PTX SHA-256: `21edf856e7e8a81f2e1b1c0ca5e31d3d52a180d0ccd67f7f8d96e7cda86c7826`.
- LLVM21 PTX SHA-256: `34a7528cdcbb6e8386c85875b860fd2ddd7e4a07cf3b277aa943a3624b657a54`.
- Validation runner SHA-256: `5fe1081cb58056d91513a265d8cdf3787af9776a333ea218fe2eef172db50dce`.
- Profiles SHA-256: `a3eaddc4a53ec0bb9b493415a5f7c940ef4b85a680f1b39546d39336cec574f4`.

PTX comes from [Actions run 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
Host CLI, logic, kernel and Cargo sources match the producer; the deliberate
build change is the CuMetal pin. Both PTX files are byte-identical to the
preceding `0242f22` replay. Every invocation logs the selected revision, paired
compiler/runtime hashes and PTX hash.

```sh
nix develop .#cumetal --command cargo build --offline --release --locked \
  -p vanity-miner --no-default-features --features cumetal,shallenge \
  --target-dir .cumetal-artifacts/shallenge-issue129-validation/target
```

Run `.cumetal-artifacts/shallenge-issue129-validation/target/release/vanity-miner`
with `--cumetal-root <matched-package>
--ptx <original-ptx> --blocks 1 --threads-per-block 32 --batches 2 --seed 1 --verify
shallenge --username <username> --target-hash <64-hex-digits>`.
Set `CUMETAL_TRACE_GPU=1` and `CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0`.

Local [metadata](../.cumetal-artifacts/shallenge-issue129-validation/metadata.json),
[profiles](../.cumetal-artifacts/shallenge-issue129-validation/profiles.json),
[exact commands and results](../.cumetal-artifacts/shallenge-issue129-validation/results.json),
[summary](../.cumetal-artifacts/shallenge-issue129-validation/summary.json),
and [runner](../.cumetal-artifacts/shallenge-issue129-validation/run.py)
retain the complete replay. Compiler regression logs are under
`../upstream-issue-breakdown/issue-129-implementation/`. These local artifacts
are not public attachments; the PR and issues contain the measured summary,
reproduction instructions and hashes.

<details>
<summary>Pre-fix measurement on 0242f22 (historical; superseded above)</summary>

### Historical Shallenge production validation on CuMetal `0242f22`

Measured **2026-09-16**, using the current locked compiler/runtime and a freshly
built normal Shallenge CLI. **LLVM21 passes; LLVM7 fails PTX translation.**
This report covers production `kernel_shallenge`, not the separate self-tests.

| PTX version | Translation / preparation | GPU and CPU verification | CLI elapsed |
| --- | --- | --- | --- |
| LLVM7 | Rejected at PTX line 306 | No GPU execution | 1.060 s |
| LLVM21 | Passed for all five profiles | 10 GPU batches, 320 candidate positions | First invocation 2.195 s; subsequent 0.886–0.902 s |

Each successful invocation ran two batches of 32 candidates with `--verify`.
Profiles: username `miner` with seeds 1 and 2 and a maximum initial target;
`miner` with a zero target (no matches); and the minimum/maximum username lengths
(1 and 30 bytes). The target tightens after a published winner, so the maximum
initial target does not require every later candidate to match.

All ten launches reported successful **Apple M5 GPU execution through generic
PTX lowering**. The CLI checked input immutability, buffer guards, CPU match/error
counts across all candidate positions, and the selected winner's full payload.
An independent Python SHA-256 check verified all seven published winners and
their decreasing hashes; every successful invocation reported 64 total nonces.
The kernel returns one winner per batch, so this does not compare every GPU
candidate's full payload or claim 320 unique inputs. No invocation timed out;
the per-invocation deadline was 600 seconds.

#### Why LLVM7 fails

The fresh diagnostic is:

```text
cumetalc failed: line 306: pointer subtraction requires a pointer minus a 64-bit integer byte offset
```

LLVM7 generates a partially unrolled nonce-filling loop from
[`generate_base64_nonce`](../logic/src/search/xoroshiro.rs). Its remaining-byte
count uses this expression, where `cursor = base + bytes_already_written`:

```text
base + (1 - cursor - username_length) + 30
  = 31 - username_length - bytes_already_written
```

The final result is a scalar count, but CuMetal rejects the intermediate
`1 - cursor` before proving that the shared buffer address cancels. The failing
instruction is `sub.s64 %rd95, %rd94, %rd172` in `kernel_shallenge`, before hashing.
LLVM21 expresses the loop with supported pointer-minus-integer arithmetic and
fixed-offset stores. Different valid PTX forms expose different compiler coverage.

The **same pinned compiler** also rejected a small reproduction of the address
expression (0.012 s) and translated its algebraically simplified scalar control
(0.012 s). These controls establish the translation gap; they are not full-kernel
GPU passes. Both production PTX files were used unchanged.

This is [cuda-metal #129](https://github.com/Lulzx/cuda-metal/issues/129), linked
from [vanity-miner #38](https://github.com/brandonros/vanity-miner-rs/issues/38).
The issue remains open, with no linked fixing PR at this check. The #76/#131
conversion and nested-pointer corrections do not implement this cancellation.
The logged compiler/runtime identities match the pin, so this failure is not
evidence of an accidental older build.

The required compiler change is to prove and simplify cancellation of the same
allocation's address, including the loop-carried cursor, while preserving
wrapping 64-bit arithmetic. It must continue rejecting unrelated pointers and
expressions where cancellation cannot be established. Acceptance needs the
small reproducer, loop/overwrite/negative cases, and an unchanged LLVM7 production
replay with CPU/GPU verification. **#38 cannot close on this result.**

#### Identity and reproduction

- Miner host: `2cd263fb1a28b784dd3967302ead48f5c72f6452`.
- PTX producer: `afe80210ea28748cc58c3ba75f877bfa4a6b1ecd`,
  [Actions run 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
  Host CLI/logic/kernel/build sources match that producer.
- CuMetal: `0242f22df09f62486e38d4c0c18a10e9098e92bd`.
- Matched package: `/nix/store/324pk362ifxdw2ckihhssf37kxn0l1k2-vanity-cumetal-0242f22df09f`.
- Compiler SHA-256: `4bdf1746c268711d74e856903c9c19d6d75fd79ce29680b6cda4ac9d15dd0b43`.
- Runtime SHA-256: `20d9b83b74684d935798d654839102ba5377cbcfa05aad6fde0f665997043c25`.
- Host CLI SHA-256: `6c05ec92655bdbe91ebbfdb9108ad65f317f96b12c3922030faae1de4aed62f5`.
- LLVM7 PTX SHA-256: `21edf856e7e8a81f2e1b1c0ca5e31d3d52a180d0ccd67f7f8d96e7cda86c7826`.
- LLVM21 PTX SHA-256: `34a7528cdcbb6e8386c85875b860fd2ddd7e4a07cf3b277aa943a3624b657a54`.

```sh
nix develop .#cumetal --command cargo build --offline --release --locked \
  -p vanity-miner --no-default-features --features cumetal,shallenge \
  --target-dir .cumetal-artifacts/shallenge-0242f22-validation/target
```

The fresh binary was run with `--cumetal-root` selecting the matched package,
`--ptx` selecting each retained input, `--blocks 1 --threads-per-block 32
--batches 2 --seed 1 --verify shallenge --username miner --target-hash <64 hex
digits>`. The other profiles change only seed, username or initial target.
`CUMETAL_TRACE_GPU=1` records execution provenance;
`CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0` selects the generic path.

Local [metadata](../.cumetal-artifacts/shallenge-0242f22-validation/metadata.json),
[results and exact commands](../.cumetal-artifacts/shallenge-0242f22-validation/results.json),
[LLVM7 log](../.cumetal-artifacts/shallenge-0242f22-validation/runs/llvm7-match-seed1/cli.log),
[LLVM21 first log](../.cumetal-artifacts/shallenge-0242f22-validation/runs/llvm21-match-seed1/cli.log),
and [diagnostic control results](../.cumetal-artifacts/shallenge-0242f22-validation/diagnostic-controls/results.json)
retain the evidence. An independent read-only audit rehashed the artifacts/logs
and verified the seven published hashes without discrepancies.

</details>
