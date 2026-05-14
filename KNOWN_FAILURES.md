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

**Live hypothesis (NEW, after slots 102/103 FAILed):** Returning a
**named-field struct wrapping `[u8; 32]` by value** is miscompiled.
Dalek's `Scalar` is `pub struct Scalar { pub(crate) bytes: [u8; 32] }`.

Evidence:
- Returns `[u8; 32]` direct: slots 70 (clamp_integer), 92 (Scalar::ONE.to_bytes) PASS.
- Returns tuple-struct wrapping `[u64; 5]` with pub field: slot 84 (`Scalar52::from_bytes`) PASS.
- Returns named-field struct wrapping `[u8; 32]` with pub(crate) field: slot 71, 102, 103 FAIL.

Bug-71 fires for ALL inputs (zero, one, anything) through BOTH entry
points (`from_bytes_mod_order`, `from_bytes_mod_order_wide`). The
common factor is the return type.

**Probe in flight:** Slot 106 — `make_wrap_named(input) -> WrapNamed`
where `WrapNamed { pub bytes: [u8; 32] }` mirrors dalek's Scalar shape.
If FAIL, that's the Bug-71 minimal repro (~15 lines).

(Slots 97, 102, 103 ruled out: const-idx Index, input-dependence, entry-point-dependence.)

**Failures owned:** 5 (slots 2, 11, 12, 71, 72) + partial 3/12.

---

### Bug-41 — base58 encoding with non-zero input

**Status.** Direct repro slots 41/42 FAIL every run. Slot 43 (all-zero
input) PASSes because the all-zero path skips the digit-extraction loop.
Every building-block primitive PASSes in isolation; bug is in an
interaction we haven't isolated.

**Slot 41 (smoking gun):**
```rust
pub fn check_base58_var_len() -> u32 {
    let input: [u8; 32] = [/* non-zero 32 bytes */];
    let mut out = [0u8; 64];
    let n = base58_encode_32(&input, &mut out);
    n == BASE58_VAR_EXPECTED.len()
        && &out[..n] == BASE58_VAR_EXPECTED
}
```

**Ruled out** (all PASS in isolation):
- Reverse range iter `(0..n).rev()` (slot 83)
- Phase A inner-mutate loop body (slot 69)
- Phase B grow-tail (slot 67)
- `dividend = carry + (limb << 32); /D; %D` shape (slot 66)
- `&'static [u64; 5]` DIVISORS reads (slots 76/77)
- `BASE58_ALPHABET` lookup (slots 60-62)

**Update:** Slot 105 FAILed → Bug-41 fires even at `limb_count == 1`.
Single non-zero byte is enough to trigger. The previous "needs multi-
iter digit extraction" hypothesis is dead.

**Probe in flight:** Slot 107 — hand-rolled `base58_encode_32` without
the `seq!` macro (which proc-macro-unrolls the outer 8-iteration loop).
Plain `while k < 8 { … }` instead. If 107 PASS but 105 FAIL, the bug is
in something specific to `seq!`'s expansion (function size, code layout,
or some shape that the proc-macro produces).

**Failures owned:** 2 direct (41, 42) + partial cascade in 3, 12, 19, 21-24.

---

### Bug-96 — `&GenericArray<u8, N>::as_slice().last()` miscompiles ✅ MINIMAL REPRO READY

**Status.** Slot 101 FAILed in v1.53.0. ~10-line minimal repro confirmed.
**Send this to compiler agent now.**

**Repro (slot 101):**
```rust
#[inline(never)]
fn last_via_as_slice(ga: &GenericArray<u8, U32>) -> u8 {
    *ga.as_slice().last().expect("non-empty")
}

pub fn check_generic_array_as_slice_last() -> u32 {
    let ga: &GenericArray<u8, U32> = (&SECP256K1_GY_BYTES).into();
    let last = last_via_as_slice(ga);
    (last == 0xB8) as u32   // SECP256K1_GY_BYTES[31] = 0xB8
}
```

**Why slot 96's whole chain fails:** `EncodedPoint::from_affine_coordinates`
calls `Tag::compress_y(y.as_slice())` to compute the parity bit. That
calls `y.as_slice().last()` on the `&GenericArray<u8, U32>` parameter.
If `last()` returns wrong byte, the tag is wrong → encoded bytes are
wrong.

**Ruled out (all PASS):**
- Slot 98 — basic GA index by fixed offset.
- Slot 99 — GA write-side `copy_from_slice` from `&[u8]` source.
- Slot 100 — same algorithm with raw `[u8; 33]` instead of GA.
- Slot 104 — `(&[u8; N]).into() → &GenericArray<u8, N>` conversion.

The narrowed shape: `GenericArray::as_slice()` (which deref-casts to
`&[T]` then takes `as_slice()`) followed by `.last()` on the result —
specifically when the GA is a `&` parameter, not a local.

**Failures owned:** 21 (slots 4, 5, 13-24, 74, 75, 78, 79, 93, 96; slot 80 now PASSes).

---

## Failures grouped by bug

Each slot is blocked by ≥1 of the three open bugs. Slots blocked by
multiple bugs need ALL of them fixed before they pass.

### Blocked by `dalek-scalar-reduce` only — 4 slots

These all execute dalek's broken `Scalar::from_bytes_mod_order` /
`reduce` chain. Once that's fixed, all four clear.

| #  | Slot name                            | Where it touches the bug |
|----|--------------------------------------|--------------------------|
| 71 | dalek scalar from-bytes round-trip   | **direct hit / smoking gun** |
| 72 | dalek mul_base scalar=1               | calls 71's path then `mul_base` |
|  2 | ed25519 derive                       | `ed25519_derive_public_key` calls `Scalar::from_bytes_mod_order` |
| 11 | solana pub                           | runs ed25519 derive (= slot 2's path) |

### Blocked by `base58-encode-32` only — 2 slots

Just the two direct probes. Once `base58_encode_32` with non-zero input
is fixed, both clear.

| #  | Slot name                            | Where it touches the bug |
|----|--------------------------------------|--------------------------|
| 41 | base58 var-len                       | **direct hit / smoking gun** |
| 42 | base58 var-len leading-zero          | variant of 41 |

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

### Compound — blocked by multiple bugs — 7 slots

These slots need MORE than one bug fixed before they pass. Fixing just
one of their blockers won't help.

| #  | Slot name                  | Blocked by                                  | Why |
|----|----------------------------|---------------------------------------------|-----|
|  3 | base58 encode pub          | dalek-scalar-reduce + base58-encode-32      | base58 of dalek's output: both the input *and* the encoder are broken |
| 12 | solana encoded             | dalek-scalar-reduce + base58-encode-32      | ed25519 → base58 in solana pipeline |
| 19 | bitcoin encoded            | k256-encodedpoint + base58-encode-32        | secp256k1 → sha+rip → base58 |
| 21 | wif compressed mainnet     | k256-encodedpoint + base58-encode-32        | secp256k1 → base58 (WIF) |
| 22 | wif uncompressed mainnet   | k256-encodedpoint + base58-encode-32        | same |
| 23 | wif compressed testnet     | k256-encodedpoint + base58-encode-32        | same |
| 24 | wif uncompressed testnet   | k256-encodedpoint + base58-encode-32        | same |

### What clears when each bug is fixed

| Fix | Slots cleared | Cumulative |
|---|---|---|
| `dalek-scalar-reduce` alone | 4 (71, 72, 2, 11) | 4 / 29 |
| `base58-encode-32` alone | 2 (41, 42) | 2 / 29 |
| `k256-encodedpoint` alone | 16 (see above) | 16 / 29 |
| dalek + base58 | 4 + 2 + 2 compound (3, 12) | 8 / 29 |
| dalek + k256 | 4 + 16 | 20 / 29 |
| base58 + k256 | 2 + 16 + 5 compound (19, 21-24) | 23 / 29 |
| **all three** | 4 + 2 + 16 + 7 compound | **29 / 29** |

So `k256-encodedpoint` is the highest-leverage fix (16 slots), followed
by `dalek-scalar-reduce` (4 direct + unlocks 2 compound). `base58-encode-32`
only directly clears 2 slots but is a prerequisite for the 7 compound failures.

---

## Active probes awaiting next run

| Slot | Targets | Probe |
|---|---|---|
| 106 | Bug-71 | named-field struct wrapping `[u8; 32]` return-by-value (mirrors dalek `Scalar`) |
| 107 | Bug-41 | hand-rolled `base58_encode_32` without `seq!` macro |

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
