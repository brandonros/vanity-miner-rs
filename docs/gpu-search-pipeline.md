# RSA and P-256 CUDA search

Each CUDA device owns its context, module, inputs, and working buffers until the
search ends. P-256 public-key derivation, both P-256 signature sources, and RSA-PSS
candidate construction and signing run in their existing device evaluators.
The host prepares each search once. It receives results through a bounded queue,
reconstructs and verifies them, and prints complete records under the stdout lock.
The GPU can process subsequent batches while verification runs. When verification
or output is slower than an easy search, the queue applies backpressure.

## RSA modulus stages

One `rsa_modulus.ptx` contains four pipeline entries, launched in order in a single
stream. Their working records remain on the device:

1. **Generate:** vacant task slots derive independent odd p candidates using
   HMAC-SHA256 and rejection sampling. Search-wide bounds and entropy are uploaded
   once. Candidate IDs never restart after a match.
2. **Construct ranges:** fixed-width division and inversion modulo a power of two
   produce the eligible q progression. Empty ranges are retired. The contiguous
   interval violating `|p-q| > 2^924` is removed from the progression. A compressed
   index skips that interval without spending candidate evaluations on it.
3. **Search:** active slots are compacted into an index list. Lanes interleave
   across that list, distributing multiple q candidates to large ranges and
   combining independent small ranges. Only actually evaluated q values are
   counted. Atomic counters retain one result per factor task; the output has
   enough slots for every active task, so there is no truncated result queue.
4. **Advance:** after the search kernel completes, each task advances by its
   assigned work. A randomized cursor wraps once through its finite progression.
   Counters remain 1024 bits on-device; large ranges are not truncated to u64.
   Exhausted and successful factor tasks are erased and replaced on later cycles.

For broad patterns, p is primality-filtered before constructing its range. At
roughly 128 constrained bytes and beyond, the order reverses: an empty range can
reject a p without expensive primality tests. The order is a scheduling heuristic;
both paths require the same factor and pattern checks.

The CPU never picks a p or prepares an individual q range in this pipeline. It
independently reconstructs p and validates completed key pairs, including fresh
random primality bases, factor separation, key consistency, and a blinded
sign/verify check. Output retires the factor task instead of exporting multiple
keys that deliberately share p.

## Bounds and measurements

`CRYPTO_BATCH_SIZE` sets both RSA workspace capacity and q-lane budget per cycle.
`THREADS_PER_BLOCK` controls every stage's launch geometry. These settings bound
memory/work per cycle, not prefix difficulty. There is no 128-byte pattern cap or
65536-q task cutoff in the CUDA pipeline. Syntax, contradictory overlap, modulus
width/parity, and the necessary factor-separation bound are still checked.

Progress reports distinguish generated p candidates, probable p factors, useful
ranges, q candidates, and independently verified outputs. Compare **verified keys
per second** under the same constraints. Counting generated p and q together is
not evidence of a speedup, and an empty range is not a tested q candidate.

Device memory is bounded by workspace capacity, approximately 1184 bytes per slot
plus fixed inputs. The host retains a bounded result queue and one retirement ID
per slot. GPU launch boundaries establish visibility between producers and
consumers; no cross-block spin loops or CPU per-factor handoffs are needed.

## Validation

Host tests compare device-compatible arithmetic against independent `BigUint`
division/inversion, including full-width patterns, empty ranges, separation holes,
large cursors, wrapping, and retirement. Queue tests cover overlapping production
and verification, cancellation, failure, and independent worker completion.
RSA self-test slots 157–159 exercise the new range, cursor, and PRF primitives
inside the separately compiled RSA self-test module.

CUDA compilation and CPU reference tests do not establish device numerical
correctness, memory safety under concurrent execution, or multi-GPU scaling.
Before throughput claims, run the rebuilt self-tests and bounded production
searches on NVIDIA hardware, then compare one and multiple GPUs. Register spills,
per-thread large-integer costs, and divergence still require measurement.

The CuMetal reference transport still uses the v2 batch interface. The new CUDA
stages have distinct entry names and require a matching host binary and PTX.

### Implementation checks — 2026-09-15

- 48 CLI unit/integration tests passed with the four crypto modes and
  `self_test_rsa_modulus` enabled, including independent RSA key verification,
  continuous searches, queue cancellation, and RSA-PSS interoperability.
- 27 logic tests and all 12 RSA self-test checks passed on the CPU.
- The aarch64 Linux release build passed with LLVM 21.1.8 and CUDA 13.3,
  producing PTX 9.3 targeting `sm_100` for all four production modules and the
  RSA self-test module. NVIDIA `ptxas` assembled all five modules successfully.
- The CuMetal host configuration passed `cargo check`; its RSA transport remains
  the reference path described above.

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
