# Known GPU self-test failures

Tracks failures from `compute-sanitizer --tool memcheck ./vanity-miner self-test`
against the current cuda-oxide alpha-NVPTX compiler. Updated after every vast run.

Self-contained: includes both our self-test layer and the downstream-crate
Rust the failing call paths execute, so this doc is enough context to hand
to the compiler agent.

## Run history

| Date       | Suite | FAIL | cuda-oxide | Notes |
|------------|-------|------|------------|-------|
| 2026-05-13 | 76    | 25   | v1.43.0    | Bug C still failing. First Bug A hypothesis. |
| 2026-05-13 | 81    | 27   | v1.46.0    | Bug C fixed. Static-array hypothesis falsified. |
| 2026-05-13 | 88    | 27   | v1.46.0    | Ladder slots 84-87 PASS — verbatim Scalar52 port works inside `logic`. |
| 2026-05-13 | 91    | 27   | v1.46.0    | Sub probes 88-90 PASS. |
| 2026-05-13 | 94    | 29   | v1.46.0    | Slot 91 FAIL (later flipped). |
| 2026-05-13 | 97    | 29   | v1.46.0    | Slot 91 PASS (Bug E lost), 96 FAIL (Bug-96 narrowed). |
| pending    | 100   | ?    | v1.52.0    | +3 probes (97-99). |
| 2026-05-13 | 100   | 28   | v1.46.0    | Slot 80 flipped PASS (silent cuda-oxide update). 97/98/99 all PASS. |
| 2026-05-14 | 100   | 28   | v1.52.0    | Identical results — 28 failures stable across 6 cuda-oxide version bumps. |
| pending    | 103   | ?    | next       | +3 probes (100-102): replica of `from_affine_coords`, GA `as_slice().last()`, dalek scalar=0 round-trip. |
| pending    | 106   | ?    | next       | +3 more probes (103-105): one per open bug — `from_bytes_mod_order_wide(0)`, `(&[u8;32]).into() &FieldBytes`, base58 single-nonzero-byte. |
| 2026-05-14 | 106   | 32   | v1.53.0    | **Breakthrough**: 101 FAIL (Bug-96 minimal repro!), 102/103 FAIL (Bug-71 fires for any input/entry-point), 105 FAIL (Bug-41 fires at limb_count=1). 100/104 PASS. |
| pending    | 108   | ?    | next       | +2 probes (106-107): named-field struct return ABI test, hand-rolled base58 without `seq!`. |
| 2026-05-14 | 108   | 32   | v1.54.0    | Slot 106 PASS (struct ABI not the bug), 107 PASS (seq! not the bug). Both new hypotheses falsified. |
| pending    | 110   | ?    | next       | +2 probes (108-109): `<[u8]>::reverse()` partial-slice, dalek Scalar `==` eq without to_bytes. |
| 2026-05-14 | 110   | 33   | v1.55.0    | **Slot 108 FAIL → Bug-41 minimal repro confirmed.** Slot 109 FAIL (Scalar value wrong). Slot 101 FLIPPED PASS (previous Bug-96 repro lost). |
| pending    | 113   | ?    | next       | +3 probes (110-112): GA-source `copy_from_slice`, `Scalar::ZERO==ZERO`, `from_canonical_bytes(0)`. |
| 2026-05-14 | 113   | 34   | v1.55.0    | **Bug-71 pinned to `reduce()` on zero**: 112 FAIL (`from_canonical_bytes([0;32])`), 111 PASS (PartialEq ok), 110 PASS (GA-source `copy_from_slice` ok; Bug-96 repro still elusive). |
| pending    | 116   | ?    | next       | +3 probes (113-115): zero-input bisect ladder — `from_bytes(0)`, `mul_internal(0, R)`, `montgomery_reduce(0)` via verbatim port. |
| 2026-05-14 | 116   | **25** | v1.56+     | **🎉 Bug-41 FIXED**: slot 108 + cascade (3, 21-24, 41, 42, 105) all PASS. **All three new zero-input probes (113/114/115) PASS** → every reduce step works on zero in our crate; real-dalek `reduce()` failing on zero is a cross-crate-composition bug. |
| pending    | 118   | ?    | next       | +2 probes (116-117): pack-of-zero (closes verbatim-on-zero loop), full reduce pipeline composed in our crate. |

---

## Currently open bugs

Three open compiler bugs. Names use the direct-hit probe's slot number —
the name points to the minimal repro.

### Bug-71 — dalek `Scalar::from_bytes_mod_order` chain

**Status.** Direct repro slot 71 FAILs every run. No isolated minimal
repro yet — slots 81-90 (ladder of every intermediate step of dalek's
`Scalar::reduce`, ported verbatim into `logic`) all PASS.

**Slot 71 (smoking gun):**
```rust
pub fn check_dalek_scalar_round_trip_one() -> u32 {
    let mut input = [0u8; 32];
    input[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(input);
    let bytes = scalar.to_bytes();
    (bytes == input) as u32   // For scalar=1, should round-trip identity
}
```

**Ruled out** (all PASS):
- `u128 >> 52` immediate shift (slot 81)
- `u128 + u128` carry chains (slot 65)
- `&'static [u64; N]` reads, raw or newtype-wrapped (slots 76/77/82)
- `Scalar52::from_bytes`, `mul_internal`, `montgomery_reduce_no_sub`, `sub`, `as_bytes` (verbatim ports — slots 84-90)
- Simple `Index<usize>`/`IndexMut<usize>` trait dispatch with `black_box(idx)` (slot 91 PASS)

**Update (v1.56+): every reduce step works on zero IN ISOLATION.**
Slots 113/114/115 ALL PASS:
- 113 — `bisect::Scalar52::from_bytes(&[0; 32]).0 == [0; 5]`
- 114 — `bisect::Scalar52::mul_internal(&ZERO, &R) == [0; 9]`
- 115 — `bisect::Scalar52::montgomery_reduce(&[0; 9]).0 == [0; 5]`

But slot 102/109/112 still FAIL (real dalek `Scalar::reduce()` on zero).
Real dalek's `reduce()` body:
```rust
fn reduce(&self) -> Scalar {
    let x = self.unpack();                       // = Scalar52::from_bytes(&self.bytes)
    let xR = Scalar52::mul_internal(&x, &Scalar52::R);
    let x_mod_l = Scalar52::montgomery_reduce(&xR);
    x_mod_l.pack()                               // = Scalar52::as_bytes(&x_mod_l)
}
```

Every step works in our crate. The bug only fires when these same steps
are composed inside dalek's crate. Strong hint: **cross-crate
inlining / monomorphization issue specific to dalek's Scalar reduce path**.

**Probes in flight:**
- Slot 116 — `Scalar52::ZERO.as_bytes() == [0; 32]`. Closes the
  every-primitive-on-zero loop (slot 87 was pack of ONE).
- Slot 117 — Full reduce pipeline composed in *our* crate on zero:
  `from_bytes → mul_internal(R) → montgomery_reduce → as_bytes`.
  - If FAIL → first Bug-71 minimal repro inside `logic`. Huge.
  - If PASS → confirmed cross-crate-only. Compiler agent will need to
    bisect dalek's monomorphization specifically.

(Slots 97, 102, 103, 106, 111 ruled out: const-idx Index, input-
dependence, entry-point-dependence, intra-crate struct-return-ABI,
PartialEq itself. Slots 113/114/115 ruled out: zero-input miscompile
of unpack, mul_internal, montgomery_reduce in isolation.)

**Failures owned (post Bug-41 fix):** 9 slots — 71, 72, 2, 11, 12, 102, 103, 109, 112.

---

### Bug-96 — `sec1::EncodedPoint::from_affine_coordinates` — minimal repro LOST

**Status.** Slot 96 still FAILs in v1.55.0, but slot 101 (our previous
minimal repro) FLIPPED to PASS. The compiler agent likely fixed the
exact `&GenericArray<u8, U32>::as_slice().last()` pattern via cuda-oxide
HEAD updates. But `from_affine_coordinates` still miscompiles —
something else inside it is the bug.

**Ruled out (all PASS):**
- Slot 98 — basic GA index by fixed offset.
- Slot 99 — GA write-side `copy_from_slice` from `&[u8]` source.
- Slot 100 — same algorithm with raw `[u8; 33]` instead of GA.
- Slot 101 — `&GA::as_slice().last()` (was FAIL in v1.53/v1.54, now PASS).
- Slot 104 — `(&[u8; N]).into() → &GenericArray<u8, N>` conversion.

**Probe in flight:** *(none for next round — Bug-71 narrowing took
priority; 113-115 are dalek-side.)* Slot 110 PASSed → `copy_from_slice`
from a `&GenericArray<u8, U32>` source works. The shape inside
`from_affine_coordinates` that breaks is still unidentified.

**Next candidates (not yet wired):**
- typenum-computed slice index: `let n = <U32 as Add<U1>>::Output::USIZE;
  dst[1..n].copy_from_slice(src)` — slot 110 used literal `1..33`
- `Tag::compress_y(y.last().expect()).into()` — enum tag computation
- `GenericArray::default()` then mutation (vs constructed via `into()`)

**Failures owned:** 21 (slots 4, 5, 13-24, 74, 75, 78, 79, 93, 96; slot 80 now PASSes).

---

## Failures grouped by bug

Each slot is blocked by ≥1 of the three open bugs. Slots blocked by
multiple bugs need ALL of them fixed before they pass.

**Current minimal repros:**
- Bug-96: **none currently** (slot 110 PASSed too; need new shape — see Bug-96 section)
- Bug-71: **slot 112** confirms `Scalar::reduce()` on zero is broken; cross-crate composition hypothesis — slots 116/117 next

### Blocked by `dalek-scalar-reduce` only — 9 slots

| #   | Slot name                            | Where it touches the bug |
|-----|--------------------------------------|--------------------------|
| 71  | dalek scalar from-bytes round-trip   | **direct hit / smoking gun** |
| 72  | dalek mul_base scalar=1              | calls 71's path then `mul_base` |
|  2  | ed25519 derive                       | `ed25519_derive_public_key` calls `Scalar::from_bytes_mod_order` |
| 11  | solana pub                           | runs ed25519 derive (= slot 2's path) |
| 12  | solana encoded                       | ed25519 → base58; base58 now works, so just dalek |
| 102 | dalek scalar round-trip ZERO          | same reduce path, zero input |
| 103 | dalek from_bytes_mod_order_wide zero  | `from_bytes_mod_order_wide` path on zero |
| 109 | dalek Scalar(0) == Scalar::ZERO       | reduce on zero, no `to_bytes` |
| 112 | dalek from_canonical_bytes(0) == ZERO | reduce via `is_canonical` |

### Blocked by `k256-encodedpoint` only — 16 slots

These all execute k256's `to_encoded_point` chain (which calls the
broken `EncodedPoint::from_affine_coordinates`). Once that's fixed, all
sixteen clear.

| #  | Slot name                            | Where it touches the bug |
|----|--------------------------------------|--------------------------|
| 96 | EncodedPoint::from_affine_coordinates| **direct hit / smoking gun** |
| 93 | k256 AffinePoint encode              | `AffinePoint::GENERATOR.to_encoded_point()` |
| 78 | k256 encode generator (no mul)       | `ProjectivePoint::GENERATOR.to_affine().to_encoded_point()` |
| 79 | k256 double generator + encode       | 78 + doubling |
| 74 | k256 derive scalar=1                 | full derive ending in `to_encoded_point` |
| 75 | k256 derive scalar=2                 | same as 74 |
| 80 | k256 Scalar::ONE round-trip          | `Scalar::to_repr` returns `GenericArray<u8, U32>` (same GA dep — likely) |
|  4 | secp256k1 compressed                 | `secp256k1_derive_public_key` calls the same chain |
|  5 | secp256k1 uncompressed               | `secp256k1_derive_public_key_uncompressed` |
| 13 | ethereum priv                        | ethereum pipeline calls secp256k1 derive |
| 14 | ethereum pub                         | same |
| 15 | ethereum address                     | same + keccak |
| 16 | bitcoin priv                         | bitcoin pipeline calls secp256k1 derive |
| 17 | bitcoin pub                          | same |
| 18 | bitcoin pkh                          | same + sha+rip |
| 20 | bitcoin matches                      | full bitcoin pipeline |

### What clears when each bug is fixed

(Bug-41 is fixed; remaining 25 failures split cleanly.)

| Fix | Slots cleared | Cumulative |
|---|---|---|
| `dalek-scalar-reduce` alone | 9 (71, 72, 2, 11, 12, 102, 103, 109, 112) | 9 / 25 |
| `k256-encodedpoint` alone | 16 (see above) | 16 / 25 |
| **both** | 25 | **25 / 25** |

`k256-encodedpoint` is the higher-leverage fix (16 slots).
`dalek-scalar-reduce` clears 9. No compound failures remain.

---

## Active probes awaiting next run

| Slot | Targets | Probe |
|---|---|---|
| 116 | Bug-71 | `Scalar52::ZERO.as_bytes() == [0; 32]` — pack-of-zero (closes verbatim-on-zero loop) |
| 117 | Bug-71 | Full reduce pipeline composed in *our* crate on zero input |

---

## Not bugs (clarifications)

- **Slot 43 (base58 all-zeros) PASS**: all-zero input → `limb_count` stays
  0 → digit-extraction loop with `DIVISORS[i]` never runs.
- **Slot 73 (`SecretKey::from_bytes(1)`) PASS** but 74/80 FAIL: 73 only
  checks validation ok-ness; doesn't inspect the inner scalar.
- **Slots 76/77 PASS** despite 41/42 FAIL: `&'static [u64; N]` reads
  work. base58 bug is not in `DIVISORS` reads.
- **Slot 65 PASS now** (was FAIL): Bug-C fixed in v1.46.

---

## History — fixed

### Bug-C (FIXED in v1.46) — u128/i128 carry chain
Slot 65 was the direct hit. Standalone upstream repro lived at
`crates/rustc-codegen-cuda/examples/i128_add_carry_chain` (commit
b6fe7ff in cuda-oxide). Fixed in v1.46.0.

### Bug-41 (FIXED in v1.56+) — `<[u8]>::reverse()` on partial sub-slice

Slot 108 was the minimal repro:
```rust
let mut arr = [0u8; 64];
arr[0] = 0x11; arr[1] = 0x22; arr[2] = 0x33; arr[3] = 0x44; arr[4] = 0x55;
let result_len = core::hint::black_box(5usize);
arr[..result_len].reverse();
// Expected: arr[0..5] = [0x55, 0x44, 0x33, 0x22, 0x11]
```

Cascade slots cleared by this fix: 3, 21, 22, 23, 24, 41, 42, 105, 108 (9 slots).
Slot 19 (bitcoin encoded) was a compound failure also blocked by Bug-96
and stays FAIL until Bug-96 is fixed.

---

## History — falsified hypotheses

Each was a live hypothesis at some point; new probe data killed it. Kept
briefly so we don't re-test the same shape.

| Hypothesis | Falsified by | What we know now |
|---|---|---|
| `&'static` multi-byte-element array reads broken (Bug A v1) | slots 76/77 PASS | `[u64; N]` static reads work, raw and depth-1 newtype-wrapped |
| `u128 >> 52` immediate right shift broken (Bug D1) | slot 81 PASS | u128 immediate shifts work |
| Depth-4 `&'static` newtype nesting broken (Bug D2) | slot 82 PASS | 4-level newtype field projections work |
| `(0..n).rev()` reverse range iterator broken (Bug D3) | slot 83 PASS | reverse range writes work |
| Simple `Index<usize>` trait dispatch broken (Bug E v1) | slot 91 FLIPPED to PASS | runtime-index Index dispatch works |
| Const-index Index trait dispatch broken (Bug E v2) | slot 97 PASS | 5-write `p[0]..=p[4]` literal-index pattern works |
| Some intermediate of dalek's Scalar reduce path miscompiles | slots 84-90 all PASS | each step is correct in isolation; bug only emerges in dalek's exact compilation |
| `Scalar52::sub` borrow-chain / volatile `black_box` broken | slots 88-90 PASS | sub works correctly |
| k256 cascade is `Lazy<>` / once_cell init failure | (not tested directly — folded into Bug-96 narrowing) | superseded by slot-96 narrowing |
| `GenericArray<u8, N>` basic index broken | slot 98 PASS | GA Deref + IndexMut at fixed offsets works |
| `GenericArray<u8, N>` write-side `copy_from_slice` broken | slot 99 PASS | dst `ga[a..b].copy_from_slice(&src[..])` works for raw `&[u8]` source |
| k256 `Scalar::ONE` round-trip broken (slot 80) | slot 80 silently flipped PASS | likely fixed by a cuda-oxide HEAD update between v1.46 and v1.52 |
| Bug-71 is input-dependent | slot 102 FAIL (all-zero input also fails) | Bug-71 fires for any input |
| Bug-71 is specific to `from_bytes_mod_order` entry point | slot 103 FAIL (`from_bytes_mod_order_wide` also fails) | Bug-71 fires for both entry points |
| Bug-96 is the basic GA Deref impl | slots 98/99/100/104 PASS, slot 101 FAIL | Bug-96 is specifically `.as_slice().last()` on `&GenericArray<T, N>` parameter |
| Bug-96 is the GA-typed parameter conversion | slot 104 PASS | `(&[u8; N]).into()` works; only `.as_slice().last()` on the result fails |
| Bug-41 needs multi-iteration digit extraction | slot 105 FAIL (limb_count=1 also fails) | Bug-41 fires for any non-zero input |
| Bug-71 is struct-return-ABI miscompile (intra-crate) | slot 106 PASS (named-field struct wrapping `[u8; 32]` returns correctly inside `logic`) | Struct-return ABI works; bug is cross-crate or in a more specific dalek path |
| Bug-41 is `seq!` macro expansion | slot 107 PASS (hand-rolled while-loop version works) | seq! isn't the bug; remaining diff is `<[u8]>::reverse()` |
| Bug-96 is `&GenericArray<u8, U32>::as_slice().last()` (slot 101) | slot 101 FLIPPED to PASS on v1.55.0 (was FAIL in v1.53/v1.54) | The compiler agent fixed THIS shape, but slot 96 still FAILs → the bug inside `from_affine_coordinates` is some other shape |
| Bug-71 is only in `to_bytes` (= `Scalar52::as_bytes`) | slot 109 FAIL (`Scalar::from_bytes_mod_order(0) != Scalar::ZERO` via PartialEq, no `to_bytes` involved) | The Scalar VALUE itself is wrong, not just byte serialization |
| Bug-71 is in dalek's `PartialEq` impl | slot 111 PASS (`Scalar::ZERO == Scalar::ZERO`) | PartialEq itself works; 109/112 failures come from wrong-value Scalar, not wrong-eq |
| Bug-96 is `dst_ga[a..b].copy_from_slice(src_ga)` | slot 110 PASS on v1.55.0 | plain copy from `&GenericArray<u8, U32>` source works; the bug is some other shape inside `from_affine_coordinates` |
| Bug-71 is `Scalar52::from_bytes` on zero | slot 113 PASS on v1.56+ | unpack of `[0; 32]` via verbatim port works |
| Bug-71 is `Scalar52::mul_internal` with a zero operand | slot 114 PASS on v1.56+ | 5×5 widening mul with one zero operand works |
| Bug-71 is `Scalar52::montgomery_reduce` on zero limbs | slot 115 PASS on v1.56+ | reduce of `[0; 9]` widened limbs via verbatim port works |

---

## Workflow

For each new vast run:

1. Update the run-history table with date / counts / cuda-oxide rev.
2. If a slot moves PASS↔FAIL, update its row in the per-failure inventory.
3. If a probe falsifies a hypothesis, move it to "History — falsified"
   with a one-row entry.
4. If a bug is fixed, move its section to "History — fixed" with the
   version tag.
5. Add new probes under the relevant "Currently open" bug section
   (and in "Active probes awaiting next run").
