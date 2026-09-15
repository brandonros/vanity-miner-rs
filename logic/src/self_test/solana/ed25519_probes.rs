//! Concrete ed25519 probes used by this mode's device self-test.
use super::*;

// Slot 70: curve25519-dalek `clamp_integer` in isolation. The smallest
// possible dalek call — pure bit-mask on bytes 0 and 31, no field math,
// no scalar repr conversion. If THIS fails, the bug is in the dalek
// API-entry plumbing itself, not in any arithmetic.
//
// Clamp definition (RFC 7748 / dalek):
//   byte[0]  &= 0xF8        (clear bits 0,1,2)
//   byte[31] &= 0x7F        (clear bit 7)
//   byte[31] |= 0x40        (set bit 6)
#[inline(never)]
pub fn check_dalek_clamp_integer() -> u32 {
    let input = core::hint::black_box([0xFFu8; 32]);
    let clamped = curve25519_dalek::scalar::clamp_integer(input);
    const EXPECTED: [u8; 32] = {
        let mut e = [0xFFu8; 32];
        e[0] = 0xF8;
        e[31] = 0x7F; // (0xFF & 0x7F) | 0x40 = 0x7F
        e
    };
    (clamped == EXPECTED) as u32
}

// Slot 71: `Scalar::from_bytes_mod_order` round-trip for the canonical
// scalar 1 (little-endian [1, 0, …, 0]). 1 is already < l, so reduction
// is a no-op and `to_bytes()` must return the input unchanged. Tests
// Scalar52 deserialization + canonical encoding without exercising any
// field multiplication or scalar mul.
#[inline(never)]
pub fn check_dalek_scalar_round_trip_one() -> u32 {
    let mut input = [0u8; 32];
    input[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(input);
    let bytes = scalar.to_bytes();
    (bytes == input) as u32
}

#[inline(never)]
pub fn check_dalek_mul_base_scalar_one() -> u32 {
    let mut scalar_bytes = [0u8; 32];
    scalar_bytes[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(scalar_bytes);
    let point = curve25519_dalek::EdwardsPoint::mul_base(&scalar);
    let compressed = point.compress().to_bytes();
    (compressed == ED25519_BASEPOINT_COMPRESSED) as u32
}

// Slot 84 — Rung A: pure byte→limbs unpack. No arithmetic, no statics
// other than the const masks.
#[inline(never)]
pub fn check_dalek_scalar52_from_bytes() -> u32 {
    let mut bytes = [0u8; 32];
    bytes[0] = 1;
    let bytes = core::hint::black_box(bytes);
    let s = bisect_scalar52::Scalar52::from_bytes(&bytes);
    (s.0 == DALEK_ONE_LIMBS) as u32
}

// Slot 85 — Rung C alone (WITHOUT the final sub call). Calls the
// `montgomery_reduce_no_sub` variant so this slot's result is directly
// comparable to the v1.46 run where it PASSed. Slot 90 calls the same
// pipeline WITH the sub — if 85 PASSes and 90 FAILs, the bug is in sub.
#[inline(never)]
pub fn check_dalek_scalar52_montgomery_reduce_r() -> u32 {
    let r = bisect_scalar52::R;
    let mut widened = [0u128; 9];
    for (i, x) in r.0.iter().enumerate() {
        widened[i] = *x as u128;
    }
    let widened = core::hint::black_box(widened);
    let result = bisect_scalar52::Scalar52::montgomery_reduce_no_sub(&widened);
    (result.0 == DALEK_ONE_LIMBS) as u32
}

// Slot 86 — Rungs B+C: mul_internal + montgomery_reduce_no_sub.
// Keeps "without sub" semantics for direct comparison to the v1.46 run.
#[inline(never)]
pub fn check_dalek_scalar52_mul_internal_then_reduce_one_r() -> u32 {
    use bisect_scalar52::Scalar52;
    const ONE: Scalar52 = Scalar52(DALEK_ONE_LIMBS);
    let one = core::hint::black_box(ONE);
    let r = core::hint::black_box(bisect_scalar52::R);
    let x_r = Scalar52::mul_internal(&one, &r);
    let result = Scalar52::montgomery_reduce_no_sub(&x_r);
    (result.0 == DALEK_ONE_LIMBS) as u32
}

// Slot 87 — Rung D: limbs→bytes pack. Inverse of slot 84.
#[inline(never)]
pub fn check_dalek_scalar52_as_bytes_one() -> u32 {
    use bisect_scalar52::Scalar52;
    const ONE: Scalar52 = Scalar52(DALEK_ONE_LIMBS);
    let one = core::hint::black_box(ONE);
    let bytes = one.as_bytes();
    let mut expected = [0u8; 32];
    expected[0] = 1;
    (bytes == expected) as u32
}

// Slot 88: `Scalar52::sub(R, R) == ZERO`. Pure borrow chain across 5 u64
// limbs, no underflow, no conditional add-L. If this FAILs, the basic
// borrow propagation is broken (Slot 47's overflowing_sub is a SINGLE
// op; this is a 5-limb chain).
#[inline(never)]
pub fn check_dalek_scalar52_sub_no_underflow() -> u32 {
    let r = core::hint::black_box(bisect_scalar52::R);
    let result = bisect_scalar52::Scalar52::sub(&r, &r);
    (result.0 == [0u64; 5]) as u32
}

// Slot 89: `Scalar52::sub(ZERO, ONE)` — triggers the underflow path.
// borrow propagates to the top bit; underflow_mask = all-1s; the
// conditional-add loop adds L back. Mathematically: 0 - 1 mod L = L - 1.
// L - 1 in 5x52-bit limbs:
//   limb[0] = 0x0002631a5cf5d3ec  (L[0] - 1)
//   limb[1..4] = L[1..4] unchanged
#[inline(never)]
pub fn check_dalek_scalar52_sub_with_underflow() -> u32 {
    use bisect_scalar52::Scalar52;
    let zero = core::hint::black_box(Scalar52::ZERO);
    let one = core::hint::black_box(Scalar52(DALEK_ONE_LIMBS));
    let result = Scalar52::sub(&zero, &one);
    let expected: [u64; 5] = [
        0x0002631a5cf5d3ec, // L[0] - 1
        0x000dea2f79cd6581, // L[1]
        0x000000000014def9, // L[2]
        0x0000000000000000, // L[3]
        0x0000100000000000, // L[4]
    ];
    (result.0 == expected) as u32
}

// Slot 90: full montgomery_reduce(widened R) with the final sub call now
// included. Compare to slot 85 (same input, sub-less version): if 85
// PASS and 90 FAIL, the bug is in `Scalar52::sub` specifically — that's
// also what makes the real dalek path (slot 71) FAIL.
#[inline(never)]
pub fn check_dalek_scalar52_montgomery_reduce_with_sub() -> u32 {
    let r = bisect_scalar52::R;
    let mut widened = [0u128; 9];
    for (i, x) in r.0.iter().enumerate() {
        widened[i] = *x as u128;
    }
    let widened = core::hint::black_box(widened);
    let result = bisect_scalar52::Scalar52::montgomery_reduce(&widened);
    (result.0 == DALEK_ONE_LIMBS) as u32
}

// Slot 92: dalek `Scalar::ONE.to_bytes()` direct. Cross-crate access to
// a `pub const Scalar` followed by trivial byte copy (Scalar's internal
// rep IS the bytes; to_bytes just copies them out). No reduce, no math.
// If this FAILs, the bug is at the cross-crate const-access layer.
#[inline(never)]
pub fn check_dalek_scalar_one_to_bytes_direct() -> u32 {
    let s = core::hint::black_box(curve25519_dalek::Scalar::ONE);
    let bytes = s.to_bytes();
    let mut expected = [0u8; 32];
    expected[0] = 1;
    (bytes == expected) as u32
}

// Slot 102: dalek `Scalar::from_bytes_mod_order([0; 32])` should
// round-trip to all-zeros (the canonical encoding of 0). Mirror of slot
// 71 with a different input value. If 102 PASS but 71 FAIL, the bug is
// input-dependent (only non-zero scalars). If 102 FAIL too, the bug is
// general to any Scalar::from_bytes_mod_order call.
#[inline(never)]
pub fn check_dalek_scalar_round_trip_zero() -> u32 {
    let input = [0u8; 32];
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(core::hint::black_box(input));
    let bytes = scalar.to_bytes();
    (bytes == input) as u32
}

// Slot 103: `Scalar::from_bytes_mod_order_wide(&[0; 64])` — uses a
// different reduction entry point than slot 71/102. Internally it calls
// `Scalar52::from_bytes_wide` + `montgomery_mul(R)` / `montgomery_mul(RR)`
// composition, NOT `Scalar::reduce`. For all-zero input the result is
// canonically 0.
#[inline(never)]
pub fn check_dalek_scalar_from_bytes_wide_zero() -> u32 {
    let input = [0u8; 64];
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order_wide(&core::hint::black_box(input));
    let bytes = scalar.to_bytes();
    (bytes == [0u8; 32]) as u32
}

// Slot 109: `Scalar::from_bytes_mod_order([0; 32]) == Scalar::ZERO`
// using dalek's `PartialEq` (which uses constant-time equality
// internally) instead of comparing the bytes output of `to_bytes`.
// Disambiguates: is the Scalar VALUE correct, or is `to_bytes` broken?
//   If 109 PASS but 102 FAIL → bug is specifically in `to_bytes`.
//   If 109 FAIL → the Scalar value from `from_bytes_mod_order` is wrong.
#[inline(never)]
pub fn check_dalek_scalar_eq_zero() -> u32 {
    use curve25519_dalek::Scalar;
    let input = core::hint::black_box([0u8; 32]);
    let s = Scalar::from_bytes_mod_order(input);
    let zero = Scalar::ZERO;
    (s == zero) as u32
}

// Slot 111: `Scalar::ZERO == Scalar::ZERO`. Pure const-vs-const
// PartialEq, no function call producing a Scalar. If FAIL, dalek's
// PartialEq impl itself is broken; if PASS, slot 109's FAIL is
// genuinely from from_bytes_mod_order returning a non-zero value.
#[inline(never)]
pub fn check_dalek_zero_eq_zero() -> u32 {
    use curve25519_dalek::Scalar;
    let a = core::hint::black_box(Scalar::ZERO);
    let b = core::hint::black_box(Scalar::ZERO);
    (a == b) as u32
}

// Slot 112: `Scalar::from_canonical_bytes([0; 32]).unwrap() == ZERO`.
// `from_canonical_bytes` does NOT call `reduce()` — it just validates
// the bytes are < ℓ and wraps. For [0; 32], 0 < ℓ so it returns
// `CtOption::Some(Scalar { bytes: [0; 32] })`. If 112 PASSes but slot
// 109 FAILs, the bug is in `reduce()` specifically (not the wider
// Scalar construction).
#[inline(never)]
pub fn check_dalek_from_canonical_zero() -> u32 {
    use curve25519_dalek::Scalar;
    let opt = Scalar::from_canonical_bytes(core::hint::black_box([0u8; 32]));
    let s_opt: Option<Scalar> = opt.into();
    let s = match s_opt {
        Some(s) => s,
        None => return 0,
    };
    let zero = Scalar::ZERO;
    (s == zero) as u32
}

// Slot 113: zero-input `from_bytes` via the verbatim Scalar52 port. Slot
// 84 covered value=1; this is the analogous zero variant. If FAIL,
// limb-unpack of all-zero bytes is broken (likely a const-fold or zero-
// special-case codegen). If PASS, the unpack step is not Bug-71's locus.
#[inline(never)]
pub fn check_dalek_scalar52_from_bytes_zero() -> u32 {
    let bytes = core::hint::black_box([0u8; 32]);
    let s = bisect_scalar52::Scalar52::from_bytes(&bytes);
    (s.0 == [0u64; 5]) as u32
}

// Slot 114: `Scalar52::mul_internal(ZERO, R)` via the verbatim port. The
// 5x5 widening multiply matrix should produce all-zero u128 limbs when
// one operand is zero. Slot 86 covered ONE * R; this is the zero case.
#[inline(never)]
pub fn check_dalek_scalar52_mul_internal_zero() -> u32 {
    use bisect_scalar52::Scalar52;
    let zero = core::hint::black_box(Scalar52::ZERO);
    let r = core::hint::black_box(bisect_scalar52::R);
    let product = Scalar52::mul_internal(&zero, &r);
    (product == [0u128; 9]) as u32
}

// Slot 115: full `montgomery_reduce(&[0; 9])` via the verbatim port —
// the final step `Scalar::reduce()` performs on a zero scalar. Slot 85
// covered widened R; slot 90 covered montgomery_reduce-with-sub on R.
// This tests the zero-input variant, which exercises the underflow-mask
// + conditional-add-L branch differently.
#[inline(never)]
pub fn check_dalek_scalar52_montgomery_reduce_zero() -> u32 {
    let widened = core::hint::black_box([0u128; 9]);
    let result = bisect_scalar52::Scalar52::montgomery_reduce(&widened);
    (result.0 == [0u64; 5]) as u32
}

// Slot 116: `Scalar52::ZERO.as_bytes() == [0; 32]` — pack of zero via
// the verbatim port. Slot 87 covered pack of ONE. Completing this
// confirms every individual reduce primitive works on zero in isolation.
#[inline(never)]
pub fn check_dalek_scalar52_as_bytes_zero() -> u32 {
    use bisect_scalar52::Scalar52;
    let zero = core::hint::black_box(Scalar52::ZERO);
    let bytes = zero.as_bytes();
    (bytes == [0u8; 32]) as u32
}

// Slot 117: full `Scalar::reduce()` pipeline composed inside our crate on
// zero input. Identical body to real dalek's `reduce()`, but every call
// resolves to the verbatim port in this same crate. If this FAILs, we
// have a Bug-71 minimal repro inside `logic` — first time. If it PASSes
// (which slots 113-115 individually doing so suggests), Bug-71 is
// genuinely cross-crate-only: only the real-dalek monomorphization of
// these same steps fires the miscompile.
#[inline(never)]
pub fn check_dalek_reduce_pipeline_zero() -> u32 {
    use bisect_scalar52::{R, Scalar52};
    let bytes = core::hint::black_box([0u8; 32]);
    let x = Scalar52::from_bytes(&bytes);
    let x_r = Scalar52::mul_internal(&x, &R);
    let reduced = Scalar52::montgomery_reduce(&x_r);
    let out = reduced.as_bytes();
    (out == [0u8; 32]) as u32
}
