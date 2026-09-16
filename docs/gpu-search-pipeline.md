# RSA and P-256 CUDA search

Each CUDA device owns its context, module, inputs, and working buffers until the
search ends. P-256 public-key derivation, both P-256 signature sources, and RSA-PSS
candidate construction and signing run in their existing device evaluators.
The host prepares each search once. It receives results through a bounded queue,
reconstructs and verifies them, and prints complete records under the stdout lock.
The GPU can process subsequent batches while verification runs. When verification
or output is slower than an easy search, the queue applies backpressure.

## RSA modulus mining

The complete mining loop lives in `logic/src/modes/rsa_modulus.rs`, a `no_std`
module compiled for CPU and GPU. The production CPU worker invokes the same
`mine` function as the thin GPU entry wrapper; only scheduling, buffers and
result publication differ. The old CPU-only BigUint search loop and its 65536-q
cutoff have been removed. BigUint remains for independent host validation and
reference arithmetic tests.

`rsa_modulus.ptx` exports only `kernel_rsa_modulus_vanity`. CUDA and CuMetal use
one launch per batch. Every lane exclusively owns a persistent `Task`; lanes do
not exchange factors or need barriers. This replaces the four-stage pipeline.

Within one launch, each lane repeats up to `--steps-per-launch` work steps:

1. An empty slot derives an odd p candidate from the secret seed and its reserved
   candidate ID. It prepares the eligible q progression and checks p's primality.
   Broad patterns check primality first; narrow patterns check the range first.
2. An active slot tests one q, advances its cursor, and retains its factor/range.
   Exhaustion erases the slot; a later step can start a new factor candidate.
3. A match publishes one pair and erases the slot immediately. That lane exits
   until the next launch, bounding output to one pair per lane even for easy
   patterns. It does not deliberately produce multiple keys sharing a factor.

The work budget defaults to 64 steps per thread, with an allowed range of 1–1024,
on both CPU and GPU.
Each step is either a p attempt or a q evaluation. It bounds work rather than
wall time: primality tests and range preparation have different costs. Raising
it reduces launch frequency but increases result/cancellation latency. A launch
boundary never discards a good p or cuts off its remaining q progression.

Each GPU launch reserves `capacity * steps` unique IDs through the shared session
counter. A new p uses `start + step * capacity + lane`; unused IDs are skipped.
An active task keeps its old ID across launches. CPU workers reserve `steps` IDs
per invocation and each retain one task with stride one. All GPUs reserve from that same
counter and maintain independent allocations and seeds. The existing bounded
host retirement table rejects repeated delivery of an already exported task.

## Arithmetic and verification

Division and inversion modulo a power of two construct compatible q values.
For an interval `[L, U]` with `U - L < p`, the shared range constructor reuses
`L = a*p + r` to calculate `floor(U/p) = a + (r + U - L >= p)`, avoiding a second
wide division. Wider intervals retain both divisions. The sum fits in 2048 bits
because both terms are below the 1024-bit factor. Compressed indices exclude the
interval violating `|p-q| > 2^924`. The randomized starting cursor traverses each
finite progression once; counts and cursors remain 1024 bits. There is no
128-byte pattern cap or 65536-q task cutoff on the GPU.

The host uploads search-wide constraints and entropy once. It never constructs
individual device q ranges. It reconstructs p for every returned pair, repeats
key validation and primality checks with fresh random bases, checks the requested
modulus pattern, and performs a blinded sign/verify before export.

CuMetal `--verify` additionally replays the complete mining invocation on the CPU
and compares every counter, returned pair and saved task. Normal execution keeps
tasks on the GPU; guard checks transfer only the allocation boundaries.

## Resources and performance

`BATCH_SIZE` is the number of persistent task slots; `THREADS_PER_BLOCK` selects
CUDA launch geometry. CuMetal capacity is `--blocks * --threads-per-block`.
Workspace is approximately 1176 bytes per slot (912-byte task and 264-byte output)
plus fixed inputs and counters. Shared statistics use atomics once per lane at
completion, outside the mining loop. No compact active list, per-task atomic
winner array, or global scheduling queue remains.

This intentionally gives up distributing one p's q work across many threads.
Independent lanes simplify ownership and resumption, but may diverge during
primality testing and require more simultaneously prepared factors. Neither
longer launches nor the refactor itself demonstrates a performance improvement.
Measure verified keys per second under identical prefix/suffix constraints on
one GPU before scaling to eight, and inspect register spills and occupancy.

The host still overlaps verification with subsequent launches through its bounded
queue. Memory and output remain bounded even when matches are frequent.

CPU and GPU progress use the shared `GlobalStats` reporter and its RSA details
extension. The `RSA stages` line shows p attempts/sec (including candidates with
no usable range), probable p factors, useful nonempty ranges, and q tests/sec.
CPU totals persist across worker restarts after a match. The main CPU rate remains
q candidates/sec; use the p rate to see work spent rejecting narrow ranges.
The p rate also includes an average per selected CPU worker or GPU device, using
the same denominator as the global rate. An explicit CPU `--threads` value is
honored. GPU totals update after completed launches; these are averages across
devices, not individual device measurements or rates per CUDA thread.

A local CPU microbenchmark of 129-byte prefix range construction measured about
6,733 ranges/sec before the division shortcut and 13,637 after (median of three
65,536-range samples, release build). This isolates range construction with
pre-generated factors; it does not measure overall mining or GPU throughput.
The manual benchmark is available with:

```sh
cargo test --release --locked --offline -p vanity-miner --no-default-features --features rsa-modulus --test rsa_device_pipeline benchmark_narrow_range_construction -- --ignored --nocapture
```

## Validation

Rebuild the host and PTX together: old four-entry PTX cannot run with this host.
Existing PTX bundles and CuMetal GPU measurements describe the prior pipeline;
they do not validate the new entry. The updated translation script requires the
expected single entry and rejects stale bundles before translation.

Host tests cover full-width range arithmetic against independent BigUint
calculations, cursor wrap/resumption, retirement, work/ID overflow, bounded result
counts, and independent verification of a pair found across two invocations.
The RSA end-to-end device self-test now exercises the same resumed mining logic.
Record fresh PTX compilation and GPU results separately from the historical
implementation checks below.

### Single-entry refactor checks — 2026-09-16

- 82 Rust tests passed: 30 logic tests (including the enabled RSA self-tests),
  41 CLI tests, and 11 RSA integration tests. The production CPU worker now uses
  the shared miner; its outputs pass independent key verification and OpenSSL
  checks. Two translation-script tests cover entry selection and report output.
- CPU-only, CuMetal, and CUDA host configurations pass `cargo check`.
- Production and RSA self-test PTX build with both LLVM 7 and LLVM 21. Each
  production artifact exports exactly `kernel_rsa_modulus_vanity`.
- CUDA 13.3 `ptxas --gpu-name sm_120 --verbose` assembles the LLVM 21 production
  kernel: **255 registers/thread, 9328-byte stack frame, 11460 bytes spill stores,
  13888 bytes spill loads, zero barriers**. These are static compiler resource
  figures, not bytes transferred per candidate or a throughput measurement.
  Register pressure and spilling are substantial; thread count alone cannot
  predict whether this implementation outperforms an eight-core CPU.
- The locked CuMetal compiler (`9e3e61574b77`) rejects the new LLVM 21 production
  PTX after 30.93 seconds: `%rd310` is undefined on an incoming edge to
  `$L__BB0_6`. It does not reach Metal compilation or GPU execution. This is a
  new-input translation observation, separate from the historical pipeline wait.
- NVIDIA GPU execution, one/eight-5090 throughput, and GPU numerical validation
  remain unmeasured. CPU/GPU search code parity does not establish GPU correctness.

Local evidence is in `.cumetal-artifacts/rsa-single-entry-20260916/`:
`source-and-ptx.json` records the consumer revision, dirty-source hashes and all
four PTX hashes; `ptxas-sm120.json` records the resource report;
`cumetal-translation.json` records the verified compiler/runtime paths, hashes,
command, timing and diagnostic log. These artifacts are intentionally outside
version control; the existing bundled PTX archives were not overwritten.

Commands used:

```sh
nix develop .#cumetal --command cargo test -p vanity-miner -p logic --no-default-features --features cumetal,rsa-modulus,self_test_rsa_modulus --release --locked --offline
python3 -m unittest discover -s scripts/tests
# In the local Linux VM:
nix develop .#v7 --command cargo check -p kernel-rsa-modulus -p kernel-self-test-rsa-modulus --release --locked --offline
nix develop .#v21 --command cargo check -p kernel-rsa-modulus -p kernel-self-test-rsa-modulus --features llvm21 --release --locked --offline
nix develop .#runner --command cargo check -p vanity-miner --no-default-features --features gpu,rsa-modulus --release --locked --offline
```

### Historical four-stage implementation checks — 2026-09-15

- 48 CLI unit/integration tests passed with the four crypto modes and
  `self_test_rsa_modulus` enabled, including independent RSA key verification,
  continuous searches, queue cancellation, and RSA-PSS interoperability.
- 27 logic tests and all 12 RSA self-test checks passed on the CPU.
- The aarch64 Linux release build passed with LLVM 21.1.8 and CUDA 13.3,
  producing PTX 9.3 targeting `sm_100` for all four production modules and the
  RSA self-test module. NVIDIA `ptxas` assembled all five modules successfully.
- The CuMetal host configuration passed `cargo check`; its RSA transport now uses the shared persistent pipeline.

The assembler reported these resources for the new production path:

| Entry | Registers/thread | Cumulative stack bytes |
| --- | ---: | ---: |
| RSA generate | 255 | 6848 |
| RSA ranges | 255 | 7424 |
| RSA search | 255 | 6896 |
| RSA advance | 168 | 400 |
| RSA-PSS | 255 | 20128 |
| P-256 public key | 255 | 3984 |
| P-256 signature | 255 | 5760 |

Every listed entry except RSA advance has register spills. These are static
assembler reports, not timings or a measured minimum `STACK_SIZE`. Parallel work
distribution is implemented; the large-integer arithmetic still needs hardware
profiling before claiming a throughput improvement.

NVIDIA runtime tests and multi-GPU measurements were unavailable: the configured
Vast SSH endpoint refused the connection. LLVM 7 / CUDA 12.9 validation was
interrupted by storage exhaustion in the build environment; the VM recovered,
but that toolchain has not passed this implementation check.
