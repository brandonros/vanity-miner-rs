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

The fixed-width range arithmetic is unchanged. Division and inversion modulo a
power of two construct compatible q values, and compressed indices exclude the
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
