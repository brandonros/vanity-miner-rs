# Solana, Bitcoin and Ethereum CuMetal validation

**All three production modes pass CPU-verified Apple GPU execution with both LLVM7 and LLVM21 PTX.**

Measured **2026-09-16** on CuMetal **`0242f22df09f62486e38d4c0c18a10e9098e92bd`**,
[draft PR #131](https://github.com/Lulzx/cuda-metal/pull/131), for
[upstream #76](https://github.com/Lulzx/cuda-metal/issues/76). The compiler fix is pushed;
the miner's updated pin and documentation remain local. On 2026-09-16, the
approved evidence was published to #23–25, upstream #76 and PR #131, and #23–25
were closed as completed. All five bodies and resulting states were read back
and verified; #76 remains open and PR #131 remains a draft.

## Results

**24 passing invocations, 48 completed GPU batches, 1,536 verified candidate positions.**
The normal miner CLI used the matched Nix compiler/runtime package, unchanged
PTX inputs, `--verify`, and a **600-second limit per invocation**. Runs were serial.
No invocation timed out. Every mode/version combination passed four profiles:
selective patterns with seeds 1 and 2, a no-match pattern, and an all-match pattern.
Each invocation ran two batches of 32 lanes, including buffer reuse.

| Mode / tracker | LLVM7 GPU + CPU | LLVM21 GPU + CPU | First LLVM7 CLI wall time | First LLVM21 CLI wall time |
| --- | --- | --- | ---: | ---: |
| [Solana #23](https://github.com/brandonros/vanity-miner-rs/issues/23) | **Pass: four profiles / eight batches** | **Pass: four profiles / eight batches** | 16.614 s | 18.027 s |
| [Bitcoin #24](https://github.com/brandonros/vanity-miner-rs/issues/24) | **Pass: four profiles / eight batches** | **Pass: four profiles / eight batches** | 346.865 s | 116.779 s |
| [Ethereum #25](https://github.com/brandonros/vanity-miner-rs/issues/25) | **Pass: four profiles / eight batches** | **Pass: four profiles / eight batches** | 315.645 s | 112.674 s |

**#23–25 are closed for this bounded production acceptance.** The separate
self-test issues and upstream #76’s remaining proof cases remain open.
The other five production modes and all eight self-test modules were not rerun;
**this is not a fresh 16-workload sweep**.

## What fixed the LLVM7 failure

At `c4e5fac`, all three LLVM7 modules emitted an invalid intermediate such as
`uchar* thread*`: the importer copied a provisional generic pointer result into
an address cast, and later legalization resolved the result without resolving
that copied inner type. A later correctly qualified load did not make the
intermediate legal Metal.

The follow-up keeps memory-address intermediates byte-typed in their actual
storage address space. Loads/stores retain their value types; the Metal emitter
constructs the final typed dereference after resolution. A recursive Metal IR
check rejects unresolved nested pointer spaces. It does not guess device storage
for generic pointers. See [implementation and focused regressions](cumetal-76-implementation.md).

## Why startup takes so long

- LLVM7 Bitcoin and Ethereum first completed in **346.865 s / 315.645 s**.
  Samples at 30, 90 and 300 seconds place both clients in Metal **pipeline creation**,
  after source-library compilation. Both eventually launched and verified.
- LLVM21 Bitcoin and Ethereum first completed in **116.779 s / 112.674 s**.
  Their 30/90-second samples also show pipeline creation.
- Later invocations took approximately **5–6 s Solana**, **17 s LLVM7 Bitcoin**,
  **16 s LLVM7 Ethereum**, and **11–12 s LLVM21 Bitcoin/Ethereum**. This is
  consistent with compilation caching; Apple’s system cache was not cleared or controlled.
- Actual GPU command-buffer intervals were **5.3–26.2 ms per batch**, measured
  by Metal `GPUEndTime - GPUStartTime`. They exclude startup compilation and are
  not isolated shader-cycle or sustained-throughput measurements.
- Fixed LLVM7 Bitcoin emits **9.41 MB** of MSL with **142,322 function-scope
  declarations**. Its search helper is 4.78 MB; `FieldElement::invert` is 2.80 MB
  and already contains about 40,383 PTX instruction starts. Inversion has no
  dispatcher cases. This is substantial inherited arithmetic plus generated
  expansion, not evidence that a single dispatcher or this qualifier fix caused
  the wait. [Static comparison](../../upstream-issue-breakdown/issue-76-llvm7-pointer-qualifiers/after-fix/timing-analysis/bitcoin-comparison.md).

Neither source size nor the client samples identifies Apple’s most expensive
compiler pass. [#127](https://github.com/Lulzx/cuda-metal/issues/127) remains a
measurable optimization candidate; this run establishes successful preparation
under the longer limit, not an optimized startup time.

## CPU and runtime checks

| Mode | Selective prefix | Selective suffix |
| --- | --- | --- |
| Solana | `E` | `y` |
| Bitcoin | `bc1ql` | `5` |
| Ethereum | `95` | `d9` |

The selective fixtures yield one match and 31 misses per batch. The CLI evaluates
all candidates on the CPU, compares aggregate match/error counts and the chosen
winner’s complete 256-byte payload, and checks unchanged request/pattern/message
bytes plus both 16-byte guards around each of four device buffers. An independent
Python oracle reconstructs printed winners’ private/public keys and addresses
and checks the final 64-candidate total. Each invocation has two successful
generic-PTX Apple M5 GPU provenance events; specialization is disabled.

CPU fixtures were retained from the previous run because relevant host/logic/kernel
sources are unchanged. Those fixtures cross-checked 384 address records against
an independent cryptographic implementation; each new CLI invocation repeats
its own CPU verification. The final independent audit also reran all 24 log checks.

**Limits:** the ABI exposes one winner per batch, not every lane’s intermediate
curve/hash outputs. The existing `seed + batch + lane` schedule repeats keys:
128 candidate positions across seeds 1 and 2 cover **34 distinct private keys per
mode**; repeated control profiles and LLVM versions do not add unique keys.
No sustained-throughput, arbitrary-input, or separate self-test claim is made.

## Exact identities and reproduction

- PTX producer: **`afe80210ea28748cc58c3ba75f877bfa4a6b1ecd`**,
  [Actions run 35055622652](https://github.com/brandonros/vanity-miner-rs/actions/runs/35055622652).
- Host base: **`a9bf1abaf961bdef4d899c3e7c4ccf5063f77719`**, plus the local pin update.
  Relevant CLI, logic, kernel, Cargo, toolchain and PTX build sources match the producer.
- Device: **Apple M5**, macOS **26.6.2 (25G83)**.
- Package: `/nix/store/324pk362ifxdw2ckihhssf37kxn0l1k2-vanity-cumetal-0242f22df09f`.
- Nix source: `/nix/store/gz8572320whk0a9nx0laadgpz45d9v7i-source`.
- Runtime: `CUMETAL_TRACE_GPU=1`, `CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0`.

| Artifact | SHA-256 |
| --- | --- |
| Compiler | `4bdf1746c268711d74e856903c9c19d6d75fd79ce29680b6cda4ac9d15dd0b43` |
| Runtime | `20d9b83b74684d935798d654839102ba5377cbcfa05aad6fde0f665997043c25` |
| Host CLI | `dcfb3620f7e5d9a54e69caacab56f54aa387c301b94e669756e6274ccef7340d` |
| inputs/llvm21/bitcoin.ptx | `4b4ed546e958f24ddbc49ec179e24eacceb40545a8114df03dadd68b28cf25c4` |
| inputs/llvm21/ethereum.ptx | `577867eb0222e0fb4922498e3cf758b300533126fa722776a7c5b854a9d404ba` |
| inputs/llvm21/solana.ptx | `61767909b9b51a30574e56525b9d5629f60256ebad43f7aa5fc06e260a3eb726` |
| inputs/llvm7/bitcoin.ptx | `f7e785f77eb67f5eee4d74cba3c4206cfa6910abf21dbba797c6f2de5fcc6a7c` |
| inputs/llvm7/ethereum.ptx | `0dd0505b3c1fa0081dfe920435b67e651f7a0dca84c289a6991b0ab5bbd213d0` |
| inputs/llvm7/solana.ptx | `d964455e49d62eb8db4761494cffa913607a51f4b778b5ce4c65f6a87010288e` |

```sh
nix build path:.#cumetal --no-link --print-out-paths
nix develop path:.#cumetal --command cargo build --offline --release --locked \
  -p vanity-miner --no-default-features --features cumetal,solana,bitcoin,ethereum

CUMETAL_TRACE_GPU=1 CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0 \
  ./vanity-miner --cumetal-root /path/to/matched-cumetal-package \
  --ptx /path/to/llvm7/bitcoin.ptx --verify --seed 1 \
  --blocks 1 --threads-per-block 32 --batches 2 \
  bitcoin-vanity --prefix bc1ql --suffix 5
```

The supervisor enforces the 600-second whole-command deadline and samples or
stops only its own process groups. All owned CLI/sample processes were reaped.
Exact commands, frozen inputs, generated MSL/ABI, samples, logs, independent
checks and the [final audit](../../upstream-issue-breakdown/issue-76-qualifier-production-validation/final-audit.json)
are retained in `../../upstream-issue-breakdown/issue-76-qualifier-production-validation/`.
These are local evidence paths, not public attachments.

## Historical comparison (`c4e5fac`)

The preceding run passed all three LLVM21 modes (first times 32.111 / 136.517 /
112.837 s) and failed all three LLVM7 modules at Apple source compilation.
The original report, hashes and 15 outcomes remain in
`../../upstream-issue-breakdown/issue-76-production-validation/`, including
`report-c4e5fac.md`. Those are historical results and do not describe the current pin.
