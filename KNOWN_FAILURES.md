# Known GPU self-test failures

Tracks failures from `compute-sanitizer --tool memcheck ./vanity-miner self-test`
against the current cuda-oxide alpha-NVPTX compiler. Updated after every vast run.

Self-contained: includes both our self-test layer (the `check_*` functions
each slot calls) and the downstream-crate Rust the call paths actually
execute, so this doc alone is enough context to hand to the compiler agent.

| Date       | Suite size | FAIL | cuda-oxide rev | Notes                                 |
|------------|------------|------|----------------|---------------------------------------|
| 2026-05-13 | 76         | 25   | v1.43.0        | 2 confirmed bugs + 1 suspected, rest cascade. |
| pending    | 81         | ?    | next           | +5 triangulation probes (76-80). |

---

## Bug A — `&'static` newtype-wrapped `[u64; N]` reads broken (dalek path)

**Status.** Hypothesis. Confirmation slots 76/77 now in suite, awaiting run.

**Hypothesis.** `&'static [u8; N]` reads work (slots 60 / 63 PASS in v1.43).
`&'static` newtype tuple struct around `[u64; N]` does not — specifically the
`Scalar52(pub [u64; 5])` shape with `Index` impl. Discriminator is element
width and/or newtype-projection through `.0`.

**Smoking gun.** Slot 71 — `Scalar::from_bytes_mod_order([1,0,…,0]).to_bytes()`
should be the identity for scalar 1 (1 < ℓ, no reduction). It FAILs. The only
non-bit-twiddle work on this path is `Scalar::reduce()` reading
`constants::L`, `constants::R`, `constants::RR` — all `Scalar52([u64; 5])`
newtypes.

### Failing slots (this bug)

- slot  2 FAIL  ed25519 derive — cascade (depends on slot 71)
- slot  3 FAIL  base58 encode pub — cascade (slot 2 output + `DIVISORS: [u64; 5]`)
- slot 11 FAIL  solana pub — cascade (slot 2)
- slot 12 FAIL  solana encoded — cascade (slot 2 + slot 3)
- slot 41 FAIL  base58 var-len — direct (reads `DIVISORS: [u64; 5]`; slot 43 PASSes because all-zero input skips that loop)
- slot 42 FAIL  base58 var-len leading-zero — direct (same as 41)
- slot 71 FAIL  dalek scalar from-bytes round-trip — **smoking gun**
- slot 72 FAIL  dalek mul_base scalar=1 == basepoint — direct (slot 71 + reads `[AffineNielsPoint; 64]` basepoint table)

### Code — our layer (logic/src/self_test.rs)

Slot 71 (smoking gun):
```rust
pub fn check_dalek_scalar_round_trip_one() -> u32 {
    let mut input = [0u8; 32];
    input[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(input);
    let bytes = scalar.to_bytes();
    (bytes == input) as u32
}
```

Slots 41 / 42 / 3 all go through:
```rust
// logic/src/base58.rs
const NEXT_LIMB_DIVISOR: u64 = 58_u64.pow(5);
const DIGITS_PER_LIMB: usize = 5;
const DIVISORS: [u64; DIGITS_PER_LIMB] = {           // ← &'static [u64; 5]
    let mut divs = [0u64; DIGITS_PER_LIMB];
    let mut val = 1u64;
    let mut i = 0;
    while i < DIGITS_PER_LIMB {
        divs[i] = val;
        val *= 58;
        i += 1;
    }
    divs
};

// later, in base58_encode_32, only runs when limb_count > 0:
for idx in (0..limb_count).rev() {
    let limb_value = limbs[idx] as u64;
    let output_offset = idx * DIGITS_PER_LIMB;
    for i in 0..DIGITS_PER_LIMB {
        let temp = (limb_value / DIVISORS[i]) % 58;  // ← read
        output[output_offset + i] = temp as u8;
    }
}
```

Slots 76 / 77 (in suite, awaiting run — minimal repros):
```rust
static STATIC_U64_TABLE: [u64; 5] = [
    0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210,
    0x1111_2222_3333_4444, 0xAAAA_BBBB_CCCC_DDDD,
    0xDEAD_BEEF_CAFE_BABE,
];

pub fn check_static_u64_array_lookup() -> u32 {
    let idx = core::hint::black_box(3usize);
    let val = STATIC_U64_TABLE[idx];
    (val == 0xAAAA_BBBB_CCCC_DDDD) as u32
}

#[repr(transparent)]
pub struct U64Wrap5(pub [u64; 5]);
static STATIC_U64_WRAPPED: U64Wrap5 = U64Wrap5([/* same 5 values */]);

pub fn check_static_struct_wrapped_u64_lookup() -> u32 {
    let idx = core::hint::black_box(3usize);
    let val = STATIC_U64_WRAPPED.0[idx];
    (val == 0xAAAA_BBBB_CCCC_DDDD) as u32
}
```

### Code — downstream (curve25519-dalek 4.1.3)

`src/scalar.rs` — entry point and reduce:
```rust
pub fn from_bytes_mod_order(bytes: [u8; 32]) -> Scalar {
    let s_unreduced = Scalar { bytes };
    let s = s_unreduced.reduce();
    debug_assert_eq!(0u8, s[31] >> 7);
    s
}

fn reduce(&self) -> Scalar {
    let x = self.unpack();
    let xR = UnpackedScalar::mul_internal(&x, &constants::R);     // ← reads R
    let x_mod_l = UnpackedScalar::montgomery_reduce(&xR);          // ← reads L
    x_mod_l.pack()
}
```

`src/backend/serial/u64/scalar.rs` — the newtype + Index + montgomery_reduce:
```rust
pub struct Scalar52(pub [u64; 5]);

impl Index<usize> for Scalar52 {
    type Output = u64;
    fn index(&self, _index: usize) -> &u64 {
        &(self.0[_index])           // ← runtime-indexed read through newtype
    }
}

pub(crate) fn montgomery_reduce(limbs: &[u128; 9]) -> Scalar52 {
    fn part1(sum: u128) -> (u128, u64) {
        let p = (sum as u64).wrapping_mul(constants::LFACTOR) & ((1u64 << 52) - 1);
        ((sum + m(p, constants::L[0])) >> 52, p)          // ← constants::L[0]
    }
    let l = &constants::L;
    let (carry, n0) = part1(        limbs[0]);
    let (carry, n1) = part1(carry + limbs[1] + m(n0, l[1]));
    let (carry, n2) = part1(carry + limbs[2] + m(n0, l[2]) + m(n1, l[1]));
    // ... 8 more lines each using l[1], l[2], l[4]
    Scalar52::sub(&Scalar52([r0, r1, r2, r3, r4]), l)
}
```

`src/backend/serial/u64/constants.rs` — the constants themselves:
```rust
pub(crate) const L: Scalar52 = Scalar52([
    0x0002631a5cf5d3ed, 0x000dea2f79cd6581,
    0x000000000014def9, 0x0000000000000000,
    0x0000100000000000,
]);
pub(crate) const LFACTOR: u64 = 0x51da312547e1b;
pub(crate) const R: Scalar52 = Scalar52([/* 5 u64s */]);
pub(crate) const RR: Scalar52 = Scalar52([/* 5 u64s */]);
```

### Next steps — cuda-oxide compiler side

1. **Confirm hypothesis at the smallest shape.** Wait for next vast run with
   slots 76 + 77.
2. **Interpret 76/77:**
   - 76 FAILs → minimal repro is bare `&'static [u64; N]`. Look at how the
     codegen emits the load — likely it's emitting an `ld.b8` (byte load)
     where an `ld.b64` is required, or the address-space tag (`.const` /
     `.global`) is wrong for the element pointer.
   - 76 PASSes, 77 FAILs → bug is in newtype projection (`.0[i]`). Field
     offset for `.0` may be mis-computed when the inner type is `[u64; N]`,
     or the codegen emits a different load width when going through the
     newtype.
   - Both PASS → size, not element width. Probe with a much larger static
     (matching the basepoint table's 6 KB).
3. **Once fixed:** rerun vast. Expect slots 2/3/11/12/41/42/71/72 all clear
   in one go (8 of the current 25 fails).

---

## Bug B (suspected) — k256 generator table or field arithmetic broken

**Status.** Suspected but distinct from Bug A. Root cause unknown.

**Why it's distinct from Bug A.** k256's generator table uses
`Lazy<[LookupTable; 33]>` (runtime-initialized via `once_cell`), **not** a
`pub const` newtype like dalek's `L`. So even if Bug A's hypothesis is
correct, it doesn't directly explain why k256 fails.

Three possible root causes, in rough order of likelihood:

1. **`Lazy<>` / `once_cell` doesn't initialize on GPU.** First-access pattern
   may not lower to anything sensible.
2. **k256 `FieldElement5x52::mul_inner` u128 carry chain.** Similar shape to
   slot 65 (Bug C) but 5-limb wide. Our slot 68 covers a 3-term chain — it
   PASSes — so this would have to be a 5-term-specific failure.
3. **Some unique k256 PTX shape we haven't isolated yet.**

### Failing slots (this bug)

- slot  4 FAIL  secp256k1 compressed — direct (k256 derive)
- slot  5 FAIL  secp256k1 uncompressed — direct (k256 derive)
- slot 13 FAIL  ethereum priv — cascade (slot 4)
- slot 14 FAIL  ethereum pub — cascade (slot 4)
- slot 15 FAIL  ethereum address — cascade (slot 4)
- slot 16 FAIL  bitcoin priv — cascade (slot 4)
- slot 17 FAIL  bitcoin pub — cascade (slot 4)
- slot 18 FAIL  bitcoin pkh — cascade (slot 4)
- slot 19 FAIL  bitcoin encoded — cascade (slot 4 + base58)
- slot 20 FAIL  bitcoin matches — cascade (slot 4)
- slot 21 FAIL  wif compressed mainnet — cascade (slot 4 + base58)
- slot 22 FAIL  wif uncompressed mainnet — cascade (slot 4 + base58)
- slot 23 FAIL  wif compressed testnet — cascade (slot 4 + base58)
- slot 24 FAIL  wif uncompressed testnet — cascade (slot 4 + base58)
- slot 74 FAIL  k256 derive scalar=1 == generator — direct
- slot 75 FAIL  k256 derive scalar=2 == 2G — direct

### Code — our layer

Slot 73 (PASS — pure validation, no field math) vs slot 74 (FAIL — full derive):
```rust
pub fn check_k256_secret_from_bytes_one() -> u32 {
    use core::mem::ManuallyDrop;
    use k256::SecretKey;
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 1;
    match SecretKey::from_bytes((&priv_bytes).into()) {
        Ok(sk) => { let _sk = ManuallyDrop::new(sk); 1 }
        Err(_) => 0,
    }
}

pub fn check_k256_derive_scalar_one() -> u32 {
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 1;
    let pub_key = secp256k1_derive_public_key(&priv_bytes);
    (pub_key == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// logic/src/secp256k1.rs — what secp256k1_derive_public_key actually does:
pub fn secp256k1_derive_public_key(private_key_bytes: &[u8; 32]) -> [u8; 33] {
    let secret_key = ManuallyDrop::new(
        SecretKey::from_bytes(private_key_bytes.into()).unwrap()
    );
    let public_key = secret_key.public_key();              // ← scalar mult
    let encoded_point = public_key.to_encoded_point(true);
    let compressed_bytes = encoded_point.as_bytes();
    let mut result = [0u8; 33];
    result.copy_from_slice(compressed_bytes);
    result
}
```

### Code — downstream (k256 0.13.4)

`src/arithmetic/mul.rs` — the `Lazy` table at the heart of mul_by_generator:
```rust
struct LookupTable([ProjectivePoint; 8]);

static GEN_LOOKUP_TABLE: Lazy<[LookupTable; 33]> =
    Lazy::new(precompute_gen_lookup_table);

fn precompute_gen_lookup_table() -> [LookupTable; 33] { /* ... */ }

fn mul_by_generator(k: &Scalar) -> ProjectivePoint {
    let table = *GEN_LOOKUP_TABLE;   // ← Lazy deref, first-access init
    /* uses table to do windowed scalar mult */
}
```

`src/arithmetic/field/field_5x52.rs` — the field math path inside scalar mult:
```rust
fn mul_inner(&self, rhs: &Self) -> Self {
    let a0 = self.0[0] as u128;   // 5 of these
    let a1 = self.0[1] as u128;
    /* ... */
    let b0 = rhs.0[0] as u128;
    /* ... */
    let m = 0xFFFFFFFFFFFFFu128;
    let r = 0x1000003D10u128;
    // 5-term widening-mul + u128 add chain similar to slot 68
    /* ~60 lines of u128 arithmetic */
}
```

### Triangulation probes (slots 78-80, in suite awaiting run)

| Slot | Probe | Tests | FAIL implies |
|---|---|---|---|
| 78 | `ProjectivePoint::GENERATOR.to_affine().to_encoded_point(true)` | projective→affine + encoding chain only, no scalar mult | encoding/serialization broken |
| 79 | `ProjectivePoint::GENERATOR.double()` then encode | one doubling + non-trivial field inversion | doubling formula or 5-wide field-mul carry chain (variant of Bug C) |
| 80 | `Scalar::ONE.to_repr() ⇄ from_repr()` | k256 Scalar (not FieldElement; wraps `U256` from crypto-bigint) | Bug A's scope wider than dalek; k256 Scalar repr hits the same shape |

### Outcome decision table

| 78 | 79 | 80 | Diagnosis |
|---|---|---|---|
| F | — | — | Encoding chain (field inversion or `to_bytes`) is broken |
| P | F | — | Doubling formula / field-mul carry chain (5-wide variant of Bug C) |
| P | P | F | k256 `Scalar` hits the same shape as Bug A |
| P | P | P | None of the above — `Lazy` (`once_cell::sync::Lazy<[LookupTable; 33]>` in `k256/arithmetic/mul.rs:367`), or scalar-mult algorithm, or table indexing. Add a direct `Lazy<u64>` probe slot if needed; it requires adding `once_cell` as a logic dep + a host-only `critical-section/std` dev-dep, so we skipped it in this round and rely on elimination. |

### Code — probes (logic/src/self_test.rs)

```rust
// Slot 78
pub fn check_k256_encode_generator() -> u32 {
    let g = k256::ProjectivePoint::GENERATOR;
    let affine = g.to_affine();
    let encoded = affine.to_encoded_point(true);
    /* compare 33 bytes to SECP256K1_GENERATOR_COMPRESSED */
}

// Slot 79
pub fn check_k256_double_generator() -> u32 {
    let g2 = k256::ProjectivePoint::GENERATOR.double();
    let affine = g2.to_affine();
    let encoded = affine.to_encoded_point(true);
    /* compare 33 bytes to SECP256K1_TWO_G_COMPRESSED */
}

// Slot 80
pub fn check_k256_scalar_one_round_trip() -> u32 {
    let s = k256::Scalar::ONE;
    let repr = s.to_repr();
    let s2 = k256::Scalar::from_repr(repr).unwrap();
    (s2 == s) as u32
}
```

### Next steps — cuda-oxide compiler side

1. **Run the suite once with all 4 probes in.** Decision table above maps
   results directly to root cause.
2. **First, see what Bug A fix does.** A k256 lazy-init bug could be masked
   by Bug A polluting constants k256 uses. Fix Bug A first; if slots
   4/5/13-24/74/75 then clear, Bug B was actually Bug A.

---

## Bug C — u128/i128 carry chain

**Status.** Filed with compiler agent. Standalone repro at
`crates/rustc-codegen-cuda/examples/i128_add_carry_chain` (commit b6fe7ff
in cuda-oxide). Awaiting fix.

### Failing slots

- slot 65 FAIL  arith i128 chain add — direct (regression coverage of the
  upstream standalone repro)

### Code — our layer

```rust
pub fn check_arith_i128_chain_add() -> u32 {
    // Sequential u128 + u128 wrapping additions that force a low→high
    // carry on every step. Mirrors the accumulation in
    // Scalar52::mul_internal and FieldElement5x52::mul_inner.
    const A: u128 = 0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFE;
    const B: u128 = 0x0000_0000_0000_0000_0000_0000_0000_0003;
    // ... full chain (see logic/src/self_test.rs)
}
```

### Next steps — cuda-oxide compiler side

1. Compiler agent has the repro.
2. After fix lands: rerun vast. Slot 65 should turn green. No cascades in
   the current suite — slots 48 / 49 / 68 cover related shapes and all
   PASS, so this pattern is genuinely isolated.

---

## Not bugs (clarifications for future readers)

- **Slot 43 (base58 all-zeros) PASSes** despite slots 41/42 FAILing because
  all-zero input causes `limb_count` to stay 0, so the DIVISORS-reading
  digit-extraction loop never executes. If Bug A is fixed and slot 43
  starts FAILing, that means we introduced a regression — slot 43's
  expected output is independent of `DIVISORS`, it's purely leading-zeros
  prepend + `BASE58_ALPHABET[0]` map + reverse.

- **Slot 73 (k256 SecretKey::from_bytes(1)) PASSes** while slot 74 FAILs.
  Correct: `from_bytes` only validates byte range; actual scalar mult and
  precomputed-table reads happen inside `public_key()`, not `from_bytes`.

- **Slot 67 (dynamic-grow stack array write) PASSes** — stack-resident
  `[u32; 10]` is fine to index dynamically. Bug A is specifically about
  `&'static` data.

---

## Workflow

For each new vast run:

1. Update the run table at the top with date / counts / cuda-oxide rev.
2. If a slot moves PASS↔FAIL, update its bullet under the relevant bug.
3. If a new independent root bug appears, add a new `## Bug X` section
   following the same structure: status, hypothesis, failing slots, our
   layer code, downstream code, next steps for the compiler side.
4. When a bug is fixed: prefix the heading with `(FIXED in vX.Y.Z)` so
   we have a historical record.
