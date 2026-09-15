//! Concrete arithmetic used by this mode's device self-test.
use super::*;

#[inline(never)]
pub fn check_arith_u32_div_var() -> u32 {
    // Two black-boxed operands — forces `div.u32` PTX op (no magic-multiply
    // folding, since the divisor isn't a known constant).
    const EXPECTED: u32 = ARITH_U32_A / 58;
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(58u32);
    (a / b == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_u32_div_const() -> u32 {
    // Variable dividend, constant divisor — rustc lowers `x / 58` to
    // `mul.hi.u32` (or `mul.wide.u32` + shift) magic-multiply. Same path
    // base58_encode_32 uses.
    const EXPECTED: u32 = ARITH_U32_A / 58;
    let a = core::hint::black_box(ARITH_U32_A);
    (a / 58 == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_u64_div_var() -> u32 {
    // Forces `div.u64` PTX op.
    const EXPECTED: u64 = ARITH_U64_A / 58;
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(58u64);
    (a / b == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_u64_div_const() -> u32 {
    // Variable dividend, constant divisor — rustc lowers `x / 58` to
    // `mul.hi.u64` (the smoking-gun op). This is THE path base58_encode_32
    // takes for its divide-by-58 reduction loop.
    const EXPECTED: u64 = ARITH_U64_A / 58;
    let a = core::hint::black_box(ARITH_U64_A);
    (a / 58 == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_u32_rem_var() -> u32 {
    // Forces `rem.u32`.
    const EXPECTED: u32 = ARITH_U32_A % 58;
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(58u32);
    (a % b == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_u64_rem_var() -> u32 {
    // Forces `rem.u64`.
    const EXPECTED: u64 = ARITH_U64_A % 58;
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(58u64);
    (a % b == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_u32_mul_lo() -> u32 {
    // Forces `mul.lo.s32` / `mul.lo.u32` (low 32 bits of u32 × u32).
    const EXPECTED: u32 = ARITH_U32_A.wrapping_mul(ARITH_U32_B);
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(ARITH_U32_B);
    (a.wrapping_mul(b) == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_u64_mul_lo() -> u32 {
    // Forces `mul.lo.s64` / `mul.lo.u64` (low 64 bits of u64 × u64). This
    // op is *heavily* used by the failing primitives but is also used by
    // some passing ones via the slice-indexing path, so it's worth a direct
    // isolated check.
    const EXPECTED: u64 = ARITH_U64_A.wrapping_mul(ARITH_U64_B);
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    (a.wrapping_mul(b) == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_u64_mul_hi() -> u32 {
    // The smoking gun: `(a as u128) * (b as u128) >> 64` lowers to a single
    // `mul.hi.u64` PTX op. Every failing primitive (ed25519 field math,
    // secp256k1 field math, base58 divide-by-constant) is dominated by
    // this exact op. If this slot FAILs on GPU and the matching CPU test
    // passes, the alpha compiler's `mul.hi.u64` codegen is broken.
    const PROD: u128 = (ARITH_U64_A as u128) * (ARITH_U64_B as u128);
    const EXPECTED: u64 = (PROD >> 64) as u64;
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let hi = (((a as u128) * (b as u128)) >> 64) as u64;
    (hi == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_u128_mul() -> u32 {
    // Full u128 wrapping multiply. Lowers to a sequence of `mul.lo.s64` +
    // `mul.hi.u64` + `mad.lo.s64`. Exercises the carry chain rustc emits
    // for >64-bit arithmetic.
    const EXPECTED: u128 = ARITH_U128_A.wrapping_mul(ARITH_U128_B);
    let a = core::hint::black_box(ARITH_U128_A);
    let b = core::hint::black_box(ARITH_U128_B);
    (a.wrapping_mul(b) == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_overflowing_add() -> u32 {
    // Three regimes in one slot so any miscompile of `add.cc.u64` /
    // `addc.cc.u64` (the PTX primitives that carry the boolean out) FAILs
    // the slot regardless of which value range trips it.
    let a = core::hint::black_box(1u64);
    let b = core::hint::black_box(2u64);
    let (s, c) = a.overflowing_add(b);
    if s != 3 || c {
        return 0;
    }

    // Carry at the wraparound boundary: u64::MAX + 1 → (0, true)
    let a = core::hint::black_box(u64::MAX);
    let b = core::hint::black_box(1u64);
    let (s, c) = a.overflowing_add(b);
    if s != 0 || !c {
        return 0;
    }

    // Saturating-style overflow: u64::MAX + u64::MAX → (u64::MAX-1, true)
    let a = core::hint::black_box(u64::MAX);
    let b = core::hint::black_box(u64::MAX);
    let (s, c) = a.overflowing_add(b);
    if s != u64::MAX - 1 || !c {
        return 0;
    }

    1
}

#[inline(never)]
pub fn check_arith_overflowing_sub() -> u32 {
    // No borrow: 5 - 3
    let a = core::hint::black_box(5u64);
    let b = core::hint::black_box(3u64);
    let (s, c) = a.overflowing_sub(b);
    if s != 2 || c {
        return 0;
    }

    // Borrow at zero boundary: 0 - 1 → (u64::MAX, true)
    let a = core::hint::black_box(0u64);
    let b = core::hint::black_box(1u64);
    let (s, c) = a.overflowing_sub(b);
    if s != u64::MAX || !c {
        return 0;
    }

    // Borrow from mid-range: 1 - u64::MAX → (2, true)
    let a = core::hint::black_box(1u64);
    let b = core::hint::black_box(u64::MAX);
    let (s, c) = a.overflowing_sub(b);
    if s != 2 || !c {
        return 0;
    }

    1
}

#[inline(never)]
pub fn check_arith_carry_chain_3limb() -> u32 {
    // Three-limb add chosen so the carry propagates through every limb.
    // [u64::MAX, u64::MAX, 0] + [1, 0, 0] = [0, 0, 1]
    // This is the literal shape dalek's Scalar52::add / k256's
    // FieldElement::add expand to, so a miscompile of the carry-propagation
    // PTX sequence (overflowing_add + boolean OR + add of `prev_carry as
    // u64`) corrupts every field-element add silently.
    let a0 = core::hint::black_box(u64::MAX);
    let a1 = core::hint::black_box(u64::MAX);
    let a2 = core::hint::black_box(0u64);
    let b0 = core::hint::black_box(1u64);
    let b1 = core::hint::black_box(0u64);
    let b2 = core::hint::black_box(0u64);

    let (s0, c0) = a0.overflowing_add(b0);
    let (s1a, c1a) = a1.overflowing_add(b1);
    let (s1, c1b) = s1a.overflowing_add(c0 as u64);
    let c1 = c1a | c1b;
    let (s2a, c2a) = a2.overflowing_add(b2);
    let (s2, c2b) = s2a.overflowing_add(c1 as u64);
    let _c2 = c2a | c2b;

    (s0 == 0 && s1 == 0 && s2 == 1) as u32
}

#[inline(never)]
pub fn check_arith_widening_mul_pair() -> u32 {
    // Tier-1 slots 38 (lo) and 39 (hi) verify each lane in isolation. This
    // one verifies both lanes come from the *same* widening product, the
    // way schoolbook multiplies in dalek/k256 consume them. A bug that
    // swaps lane association passes 38 and 39 individually but FAILs here.
    const PROD: u128 = (ARITH_U64_A as u128) * (ARITH_U64_B as u128);
    const EXPECTED_LO: u64 = PROD as u64;
    const EXPECTED_HI: u64 = (PROD >> 64) as u64;

    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let p = (a as u128) * (b as u128);
    let lo = p as u64;
    let hi = (p >> 64) as u64;
    (lo == EXPECTED_LO && hi == EXPECTED_HI) as u32
}

#[inline(never)]
pub fn check_arith_mad_lo_u64() -> u32 {
    // `a.wrapping_mul(b).wrapping_add(c)` typically folds to a single
    // `mad.lo.u64` PTX op. This is dalek's `m!` macro shape — slot 38 only
    // tests the mul, so a MAD-folding-only codegen bug slips through.
    const EXPECTED: u64 = ARITH_U64_A
        .wrapping_mul(ARITH_U64_B)
        .wrapping_add(ARITH_U64_A);
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let c = core::hint::black_box(ARITH_U64_A);
    (a.wrapping_mul(b).wrapping_add(c) == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_mad_hi_u64() -> u32 {
    // High-half MAD: same shape as mad_lo but pulling the upper 64 bits
    // of the widening product before the add. May fold to `mad.hi.u64`.
    const PROD: u128 = (ARITH_U64_A as u128) * (ARITH_U64_B as u128);
    const EXPECTED: u64 = ((PROD >> 64) as u64).wrapping_add(ARITH_U64_A);

    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let c = core::hint::black_box(ARITH_U64_A);
    let hi = (((a as u128) * (b as u128)) >> 64) as u64;
    (hi.wrapping_add(c) == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_mul_wide_u32() -> u32 {
    // Both operands start as u32 then widen to u64 for the mul — rustc may
    // emit `mul.wide.u32` (one PTX op, distinct from `mul.lo.u64`). k256's
    // 32-bit big-int paths take exactly this shape.
    const EXPECTED: u64 = (ARITH_U32_A as u64) * (ARITH_U32_B as u64);
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(ARITH_U32_B);
    ((a as u64) * (b as u64) == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_mask_blend_true() -> u32 {
    // The subtle::Choice / CtOption idiom: bool → u64 → wrapping_neg gives
    // an all-1s or all-0s mask; (a & mask) | (b & !mask) selects a or b.
    // k256's `SecretKey::from_bytes(...).unwrap()` runs through CtOption
    // whose unwrap is a const-time select on the validity flag — if the
    // `cond as u64` → `wrapping_neg()` lowering is wrong, unwrap silently
    // returns the wrong arm (matches the "consistent-but-wrong" secp256k1
    // symptom).
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let cond = core::hint::black_box(true);
    let mask = (cond as u64).wrapping_neg();
    let r = (a & mask) | (b & !mask);
    (r == ARITH_U64_A) as u32
}

#[inline(never)]
pub fn check_arith_mask_blend_false() -> u32 {
    // Same as above but with cond=false — selects b. Splitting true/false
    // into two slots means a bug that breaks only one arm pinpoints
    // immediately.
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(ARITH_U64_B);
    let cond = core::hint::black_box(false);
    let mask = (cond as u64).wrapping_neg();
    let r = (a & mask) | (b & !mask);
    (r == ARITH_U64_B) as u32
}

#[inline(never)]
pub fn check_arith_var_shr_u64() -> u32 {
    // Runtime shift amount — emits `shr.b64 %rd, %rd, %r` (variable form),
    // distinct from constant-amount shifts which can be folded. Montgomery
    // reductions in k256 do variable shifts during scalar splitting.
    const EXPECTED: u64 = ARITH_U64_A >> 13;
    let a = core::hint::black_box(ARITH_U64_A);
    let n = core::hint::black_box(13u32);
    (a >> n == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_var_shl_u64() -> u32 {
    // Same as var_shr but the other direction (`shl.b64`).
    const EXPECTED: u64 = ARITH_U64_A << 13;
    let a = core::hint::black_box(ARITH_U64_A);
    let n = core::hint::black_box(13u32);
    (a << n == EXPECTED) as u32
}

#[inline(never)]
pub fn check_arith_blackbox_identity_u64() -> u32 {
    // The cheapest possible probe: does black_box preserve a u64?
    // No arithmetic of any kind — if this FAILs on GPU + PASSes on CPU,
    // black_box's PTX lowering doesn't preserve the value, and every
    // tier-1/tier-2 arith slot's FAIL is a black_box artifact, not an op
    // bug. Tests with `0xDEADBEEFCAFEBABE` so a zero return is obviously
    // wrong.
    let v: u64 = 0xDEADBEEFCAFEBABE;
    (core::hint::black_box(v) == v) as u32
}

#[inline(never)]
pub fn check_arith_blackbox_identity_u32() -> u32 {
    // u32 variant — same probe at half the width in case the bug is
    // type-specific.
    let v: u32 = 0xDEADBEEF;
    (core::hint::black_box(v) == v) as u32
}

#[inline(never)]
pub fn check_arith_divrem_by_58_pow_5() -> u32 {
    // Slot 59 covers `x / 58` through `x / 58^4`. base58_encode_32's limb
    // update loop divides by `58^5 = 656_356_768` (NEXT_LIMB_DIVISOR),
    // which produces a *different* magic-multiply constant than slot 59
    // exercises (PTX inspection of v1.42 confirms: 7_544_311_872_078_572_213
    // for /58^5, distinct from slot 59's constants). This slot covers
    // exactly that gap.
    //
    // Cases include the first 4 bytes of slot 3's input, the divisor
    // itself, divisor-1 boundary, a known quotient with non-zero
    // remainder, and u64 extremes.
    //
    // Layout note: parallel primitive arrays (INPUTS / EXPECTED_Q /
    // EXPECTED_R) instead of an `[(u64, u64, u64); N]` array of tuples.
    // cuda-oxide as of 5feaf2e doesn't handle tuple-element array
    // constants (`translate_array_constant` only takes the integer-element
    // branch); parallel arrays of primitives go through the supported
    // path. Semantics are identical to the tuple form.
    const D: u64 = 58_u64.pow(5);
    const INPUTS: [u64; 6] = [
        0x089A23FF,
        D,
        D - 1,
        D.wrapping_mul(7).wrapping_add(123),
        u64::MAX,
        0xFFFFFFFF_00000000,
    ];
    const EXPECTED_Q: [u64; 6] = [
        0x089A23FF_u64 / D,
        1,
        0,
        7,
        u64::MAX / D,
        0xFFFFFFFF_00000000_u64 / D,
    ];
    const EXPECTED_R: [u64; 6] = [
        0x089A23FF_u64 % D,
        0,
        D - 1,
        123,
        u64::MAX % D,
        0xFFFFFFFF_00000000_u64 % D,
    ];

    let mut i = 0;
    while i < INPUTS.len() {
        let x = core::hint::black_box(INPUTS[i]);
        if x / D != EXPECTED_Q[i] || x % D != EXPECTED_R[i] {
            return 0;
        }
        i += 1;
    }
    1
}

#[inline(never)]
pub fn check_arith_i128_chain_add() -> u32 {
    // Slot 40 (u128 wrapping_mul) and slot 49 (widening mul pair) both
    // PASS, but those only exercise a single u128 op. dalek's
    // Scalar52::mul_internal and k256's FieldElement5x52::mul_inner
    // accumulate ~25 widening products via sequential `u128 + u128`
    // chains. Each addition must propagate the low-half carry into the
    // high half. If that's broken, every accumulation step silently
    // drops a bit — which would explain the residual failure of slots
    // 2/4/5/11–20 even after the overflowing_add fix.
    //
    // Three regimes per the cuda-oxide divrem_large_const_repro: pure
    // low→high carry, carry-rolls-fully-over (both halves saturated),
    // and combined low+high adds.

    // Case 0: (MAX, 0) + (1, 0) = (0, 1). Pure low→high carry.
    {
        let a = core::hint::black_box(u64::MAX as u128);
        let b = core::hint::black_box(1u128);
        const E: u128 = (u64::MAX as u128).wrapping_add(1);
        if a.wrapping_add(b) != E {
            return 0;
        }
    }

    // Case 1: (MAX, MAX) + (1, 0) = (0, 0). Carry rolls all the way over.
    {
        let a = core::hint::black_box(u128::MAX);
        let b = core::hint::black_box(1u128);
        const E: u128 = u128::MAX.wrapping_add(1);
        if a.wrapping_add(b) != E {
            return 0;
        }
    }

    // Case 2: 4-operand chain forcing carry on every step.
    // (MAX_LO) + (MAX_LO) + (MAX_LO) + ((1 << 64) | 1)
    //   → low halves wrap three times (3 carries to high) + 1 from d's
    //     high half → high = 4, low = ((MAX*3) wrapping) + 1.
    {
        let a = core::hint::black_box(u64::MAX as u128);
        let b = core::hint::black_box(u64::MAX as u128);
        let c = core::hint::black_box(u64::MAX as u128);
        let d = core::hint::black_box((1u128 << 64) | 1u128);
        let s = a.wrapping_add(b).wrapping_add(c).wrapping_add(d);
        const E: u128 = (u64::MAX as u128)
            .wrapping_add(u64::MAX as u128)
            .wrapping_add(u64::MAX as u128)
            .wrapping_add((1u128 << 64) | 1u128);
        if s != E {
            return 0;
        }
    }

    1
}

#[inline(never)]
pub fn check_arith_widening_mul_chain_3term() -> u32 {
    // dalek's `Scalar52::mul_internal` and k256's
    // `FieldElement5x52::mul_inner` accumulate partial products as
    //   z = m(a0, b2) + m(a1, b1) + m(a2, b0)
    // where m(x, y) = `(x as u128) * (y as u128)`. Slot 40 (one u128
    // wrapping_mul) and slot 49 (one widening pair) both PASS; slot
    // 65 covers a chain of u128 adds. This composes them: three
    // widening mults summed via `u128 + u128 + u128`. If a register-
    // pressure or scheduling bug only surfaces under composition,
    // this slot catches it where the isolated ones don't.
    //
    // 52-bit operands (the actual limb size dalek uses) → products
    // span ~104 bits, sums span ~106, forcing real carries across
    // the 64-bit boundary in every step.
    const A0: u64 = 0x000F_FFFF_FFFF_FFFF;
    const A1: u64 = 0x000F_FFFF_FFFF_FFFE;
    const A2: u64 = 0x000F_FFFF_FFFF_FFFD;
    const B0: u64 = 0x000F_FFFF_FFFF_FFFC;
    const B1: u64 = 0x000F_FFFF_FFFF_FFFB;
    const B2: u64 = 0x000F_FFFF_FFFF_FFFA;
    const EXPECTED: u128 = (A0 as u128)
        .wrapping_mul(B2 as u128)
        .wrapping_add((A1 as u128).wrapping_mul(B1 as u128))
        .wrapping_add((A2 as u128).wrapping_mul(B0 as u128));

    let a0 = core::hint::black_box(A0) as u128;
    let a1 = core::hint::black_box(A1) as u128;
    let a2 = core::hint::black_box(A2) as u128;
    let b0 = core::hint::black_box(B0) as u128;
    let b1 = core::hint::black_box(B1) as u128;
    let b2 = core::hint::black_box(B2) as u128;

    let z = a0
        .wrapping_mul(b2)
        .wrapping_add(a1.wrapping_mul(b1))
        .wrapping_add(a2.wrapping_mul(b0));
    (z == EXPECTED) as u32
}

// Slot 81: `u128 >> 52` immediate right shift, matching the exact shape
// inside dalek's `montgomery_reduce::part1`:
//   ((sum + m(p, constants::L[0])) >> 52, p)
// LLVM lowers u128 immediate shifts to multi-step 64-bit shift sequences.
// Slot 65 (i128 add chain) is fixed but doesn't cover this shape. Slot
// 55/56 cover u64 var shifts, not u128 immediate shifts.
#[inline(never)]
pub fn check_arith_u128_imm_shr_52() -> u32 {
    const SUM: u128 = 0xFEDC_BA98_7654_3210_0123_4567_89AB_CDEF;
    const EXPECTED: u128 = SUM >> 52;
    let sum = core::hint::black_box(SUM);
    let shifted = sum >> 52;
    (shifted == EXPECTED) as u32
}
