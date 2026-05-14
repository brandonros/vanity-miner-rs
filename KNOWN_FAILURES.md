# Known GPU self-test failures

Tracks failures from `compute-sanitizer --tool memcheck ./vanity-miner self-test`
against the current cuda-oxide alpha-NVPTX compiler. Updated after every vast run.

Self-contained: includes both our self-test layer (the `check_*` functions
each slot calls) and the downstream-crate Rust the call paths actually
execute, so this doc alone is enough context to hand to the compiler agent.

| Date       | Suite size | FAIL | cuda-oxide rev | Notes                                 |
|------------|------------|------|----------------|---------------------------------------|
| 2026-05-13 | 76         | 25   | v1.43.0        | First Bug A hypothesis (now falsified). |
| 2026-05-13 | 81         | 27   | v1.46.0        | Bug C fixed. Bug A falsified by 76/77. Re-bisecting via slots 81-83. |
| pending    | 88         | ?    | v1.46 (rerun)  | +3 hypothesis probes (81-83) and +4 ladder rungs (84-87). |
| 2026-05-13 | 88         | 27   | v1.46.0        | All 81-87 PASSed. New finding: ladder copy missing `Scalar52::sub` from montgomery_reduce. |
| pending    | 91         | ?    | next           | +3 sub probes (88-90) added; 85/86 kept on `_no_sub` variant for direct comparison. |

---

## (FIXED in v1.46) Bug C — u128/i128 carry chain

**Status.** ✅ Fixed in v1.46.0. Slot 65 PASSes.

Kept here for historical context. Standalone repro lived at
`crates/rustc-codegen-cuda/examples/i128_add_carry_chain` (commit b6fe7ff in
cuda-oxide).

---

## (FALSIFIED) Bug A v1 — `&'static` newtype-wrapped `[u64; N]` reads

**Status.** ❌ Hypothesis falsified. Slot 76 (`&'static [u64; 5]` indexed
read) and slot 77 (`&'static StructWrap([u64; 5])` indexed read) both PASS
in v1.46. The "element width > 1 byte breaks static reads" theory is dead.

Slots 71, 41, 42 still FAIL, so the actual root cause is elsewhere.

---

## Bug D (new) — three competing hypotheses for slot 71 / 41 / 42

**Status.** Three triangulation probes added (slots 81-83). Next vast run
selects between them.

### What we still know

- Slot 71 — `Scalar::from_bytes_mod_order([1,0,…,0]).to_bytes() == input` FAILs
- Slot 41/42 — base58 with non-zero input FAILs (slot 43 with all-zero PASSes)
- Slot 65 (i128 chain add) PASSes. So pure u128 add chains work.
- Slot 68 (3-term widening mul + add chain) PASSes. So that shape works.
- Slot 76/77 (`&'static [u64; N]` reads) PASS. So static reads work.

### Candidate root causes (testing now)

#### Hypothesis D1 — `u128 >> 52` immediate right shift miscompiles

Inside dalek's `montgomery_reduce::part1`:
```rust
((sum + m(p, constants::L[0])) >> 52, p)
```
LLVM lowers `u128 >> imm` to a multi-step 64-bit shift sequence. Slot 55/56
cover u64 var shifts, slot 65 covers pure u128 add — neither covers this.

**Probe:** Slot 81 — `check_arith_u128_imm_shr_52()`.

If 81 FAILs, this is the one. Explains slot 71 (dalek
montgomery_reduce), slots 78/79 (k256 `mul_inner` reduction uses
the same shape), and the cascades.

#### Hypothesis D2 — Deeper `&'static` newtype nesting (depth > 2) broken

k256's `Scalar::ONE` is:
```rust
Scalar(U256)
  U256 = Uint<4> { limbs: [Limb; 4] }
    Limb(u64)
```
Accessing inner u64 = `scalar.0.limbs[i].0` = depth-4 GEP. Slot 77 only
tested depth-2 (`Wrap([u64; 5])`).

**Probe:** Slot 82 — `check_static_depth4_newtype_nesting()` with
`ProbeScalar(ProbeUint4 { limbs: [ProbeLimb(u64); 4] })`.

If 81 PASSes and 82 FAILs, this is it. Explains slot 80 directly.

#### Hypothesis D3 — Reverse range iterator `(0..N).rev()` broken

base58's digit-extraction loop is the one shape slots 60-69 don't cover:
```rust
for idx in (0..limb_count).rev() {
    let output_offset = idx * DIGITS_PER_LIMB;
    output[output_offset + i] = ...;
}
```

**Probe:** Slot 83 — `check_reverse_range_write()`.

If 81/82 PASS and 83 FAILs, this is base58's specific bug — separate from
slot 71's bug (which would need yet another probe).

### Decision table (next vast run)

| 81 | 82 | 83 | Diagnosis |
|---|---|---|---|
| F | — | — | u128 imm shr broken. Likely explains 71/78/79 (modular reduction shape) and the dalek/k256 cascades. base58 41/42 may still need 83's signal. |
| P | F | — | Deeper static-newtype nesting broken. Explains 80 directly; 71 (Scalar52 is depth-2 so wouldn't be hit) needs another probe. |
| P | P | F | Reverse range iterator broken. Explains 41/42 base58 digit extraction. Doesn't explain 71/80. |
| P | P | P | None of these — need fresh probe round. |

### Code — our layer

```rust
// Slot 81
pub fn check_arith_u128_imm_shr_52() -> u32 {
    const SUM: u128 = 0xFEDC_BA98_7654_3210_0123_4567_89AB_CDEF;
    const EXPECTED: u128 = SUM >> 52;
    let sum = core::hint::black_box(SUM);
    let shifted = sum >> 52;
    (shifted == EXPECTED) as u32
}

// Slot 82
#[repr(transparent)] pub struct ProbeLimb(pub u64);
#[repr(C)]           pub struct ProbeUint4 { pub limbs: [ProbeLimb; 4] }
#[repr(transparent)] pub struct ProbeScalar(pub ProbeUint4);

static NESTED_ONE_PROBE: ProbeScalar = ProbeScalar(ProbeUint4 {
    limbs: [ProbeLimb(...), ProbeLimb(...), ProbeLimb(...), ProbeLimb(...)],
});

pub fn check_static_depth4_newtype_nesting() -> u32 {
    let idx = core::hint::black_box(2usize);
    let v = NESTED_ONE_PROBE.0.limbs[idx].0;  // depth-4 field projection
    (v == 0x9999_AAAA_BBBB_CCCC) as u32
}

// Slot 83
pub fn check_reverse_range_write() -> u32 {
    let limb_count: usize = core::hint::black_box(3);
    let mut out = [0u32; 10];
    for idx in (0..limb_count).rev() {
        out[idx] = (idx as u32) * 100;
    }
    // compare to const-eval baseline
}
```

### Code — downstream (where each candidate would bite)

D1 — dalek `src/backend/serial/u64/scalar.rs`:
```rust
fn part1(sum: u128) -> (u128, u64) {
    let p = (sum as u64).wrapping_mul(constants::LFACTOR) & ((1u64 << 52) - 1);
    ((sum + m(p, constants::L[0])) >> 52, p)   // ← u128 >> 52
}
```

D2 — k256 `src/arithmetic/scalar.rs` + crypto-bigint:
```rust
pub const ONE: Self = Self(U256::ONE);                  // Scalar
// where U256 = Uint<4> { pub(crate) limbs: [Limb; 4] }
// where Limb = pub struct Limb(pub Word)               // Word = u64
```

D3 — `logic/src/base58.rs:73`:
```rust
for idx in (0..limb_count).rev() {                      // ← reverse range
    let limb_value = limbs[idx] as u64;
    let output_offset = idx * DIGITS_PER_LIMB;
    for i in 0..DIGITS_PER_LIMB {
        let temp = (limb_value / DIVISORS[i]) % 58;
        output[output_offset + i] = temp as u8;
    }
}
```

### Next steps — cuda-oxide compiler side

1. Run vast with slots 81-83 in suite.
2. Decision table above maps result directly to root cause.
3. If 81 FAIL: hand slot 81 source as the minimal repro for u128 immediate
   shift. Codegen likely emits a `shr` sequence on the i128 split that
   gets the carry/upper bits wrong.
4. If 82 FAIL: hand slot 82 source. Codegen likely mis-computes GEP offsets
   when going through nested `#[repr(transparent)]` field projections.
5. If 83 FAIL: hand slot 83. Likely `core::ops::Range::next_back` (used by
   `Rev<Range>`) doesn't lower correctly — either the decrement-then-yield
   pattern or how `DoubleEndedIterator` dispatches.

### Ladder bisect (slots 84-87) — falls back to step-by-step if 81-83 are inconclusive

dalek's `backend` module is `pub(crate)`, so we **verbatim-copied** the
relevant Scalar52 code into [`logic/src/self_test.rs`](logic/src/self_test.rs)
under `mod bisect_scalar52`. This lets us call each step in isolation
without touching dalek API. The copy includes: `Scalar52` struct,
`from_bytes`, `as_bytes`, `mul_internal`, `montgomery_reduce`, plus the
`L`, `LFACTOR`, `R` constants. Same Rust source, just inside our crate.

| Slot | Rung | Tests | If FAIL |
|---|---|---|---|
| 84 | A | `Scalar52::from_bytes(&[1,0,…])` returns `[1,0,0,0,0]` | byte→limb bit-shift/OR loop broken |
| 85 | C alone | `Scalar52::montgomery_reduce(widened R as [u128;9])` returns `[1,0,0,0,0]` | montgomery_reduce broken (u128 chain + L reads + `u128>>52`) |
| 86 | B+C | `Scalar52::montgomery_reduce(mul_internal(ONE, R))` returns `[1,0,0,0,0]` | mul_internal 5×5 widening matrix broken (if 85 PASS), else cascade |
| 87 | D | `Scalar52([1,0,0,0,0]).as_bytes()` returns `[1,0,…]` | limb→byte pack broken |

Outcome decoding (subset, assuming 71 still FAILs in the run):
- 84 P / 85 P / 86 P / 87 P → bug isn't in dalek's Scalar52 math at all; must be in `Scalar` wrapper or `from_bytes_mod_order` plumbing
- 84 P / 85 F → montgomery_reduce broken; report part1/part2 lowering
- 84 P / 85 P / 86 F → mul_internal broken; report 5×5 partial-product matrix
- 84 F or 87 F → byte/limb bit packing broken; trivial PTX repro

Combined with slots 81-83 results, this should pin the bug to a single
function with a known input.

### Ladder-round-1 result (v1.46 rerun)

All four ladder rungs PASSed. Re-examining the port revealed that my
copy of `montgomery_reduce` **dropped the final `Scalar52::sub(result, L)`
call** that dalek's real implementation ends with. That meant the ladder
was technically incomplete — `sub` is part of montgomery_reduce in dalek
but wasn't exercised by any of slots 84-87.

### Sub probes (slots 88-90) — added after ladder round 1

Adding `Scalar52::sub` to the port + 3 probes:

| Slot | Probe | Tests | If FAIL |
|---|---|---|---|
| 88 | `Scalar52::sub(R, R) == 0` | 5-limb borrow chain, no underflow | basic borrow propagation broken |
| 89 | `Scalar52::sub(ZERO, ONE)` underflow → `L - 1` | borrow chain + underflow_mask + conditional-add-L (volatile `black_box`) | underflow-path conditional add broken |
| 90 | `montgomery_reduce(widened_R)` *with* final sub | same as slot 85 but calls `montgomery_reduce` (with sub) instead of `montgomery_reduce_no_sub` | sub is what's broken in real dalek path |

Slots 85 and 86 now explicitly use `montgomery_reduce_no_sub` so their
result preserves the v1.46 PASSing-meaning. Comparing slot 85 (passes,
no sub) vs slot 90 (same input, with sub) directly isolates whether sub
is the bug.

Outcome decoding:
- 85 P / 88 F or 89 F / 90 F → `Scalar52::sub` is the bug. Likely culprit:
  the volatile-load `black_box` (cuda-oxide may be miscompiling
  `core::ptr::read_volatile` to nvptx) or the 5-limb borrow chain.
- 85 P / 88 P / 89 P / 90 F → the bug isn't in sub itself, but in the
  *sequencing* of mul_internal → reduce → sub (some pass-pipeline issue).
- 85 P / 88 P / 89 P / 90 P / 71 still F → it's truly cross-crate. Bug
  manifests when the same code is compiled inside `curve25519-dalek` but
  not when compiled inside `logic`. Time to spin up a separate-crate
  experiment.

---

## Remaining open failures (mapped to candidate root causes)

After v1.46.0 run:

| Slot | Name | Suspected root cause | Notes |
|---|---|---|---|
|  2 | ed25519 derive | D1 or D2 | cascade of slot 71 |
|  3 | base58 encode pub | D1 + D3 | cascade of slot 2 + base58 issue |
|  4 | secp256k1 compressed | D1 or D2 | cascade of slot 78 |
|  5 | secp256k1 uncompressed | D1 or D2 | cascade of slot 78 |
| 11-12 | solana pub/encoded | D1 or D2 | cascade |
| 13-15 | ethereum | D1 or D2 | cascade |
| 16-24 | bitcoin / WIF | D1 or D2 + D3 | cascade |
| 41 | base58 var-len | D3 (most likely) | direct |
| 42 | base58 var-len leading-zero | D3 | direct |
| 71 | dalek scalar round-trip | D1 (most likely) | direct |
| 72 | dalek mul_base scalar=1 | D1 | direct, depends on 71 |
| 74 | k256 derive scalar=1 | D1 or D2 | direct |
| 75 | k256 derive scalar=2 | D1 or D2 | direct |
| 78 | k256 encode generator | D1 | direct |
| 79 | k256 double generator | D1 | direct |
| 80 | k256 Scalar round-trip | D2 (most likely) | direct |

---

## Not bugs (clarifications for future readers)

- **Slot 43 (base58 all-zeros) PASSes** despite slots 41/42 FAILing because
  all-zero input causes `limb_count` to stay 0, so the digit-extraction
  loop with `(0..limb_count).rev()` and `DIVISORS[i]` access never runs.

- **Slot 73 (k256 SecretKey::from_bytes(1)) PASSes** while slot 74/80 FAIL.
  Correct: `from_bytes` returns Result, not CtOption, and its caller only
  inspects ok-ness, not the inner scalar's bytes.

- **Slot 76/77 PASS** despite slot 41/42 FAILing. Falsifies the static-
  multi-byte-array hypothesis. The base58 bug is not in the DIVISORS read.

- **Slots 65, 68 PASS** (u128 add chain, 3-term widening). Falsifies the
  "wide u128 chain" hypothesis at that depth. If D1 confirms, the issue is
  specifically immediate-amount shifts on u128, not add or mul chains.

---

## Workflow

For each new vast run:

1. Update the run table at the top with date / counts / cuda-oxide rev.
2. If a slot moves PASS↔FAIL, update its row in the open-failures table.
3. If a hypothesis is falsified, move its section under a `## (FALSIFIED)`
   heading and keep it for historical context.
4. If a hypothesis is confirmed and fixed, rename to `## (FIXED in vX.Y.Z)`.
