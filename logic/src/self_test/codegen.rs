//! Arithmetic and compiler regression checks.
use crate::{
    crypto::secp256k1::secp256k1_derive_public_key, encoding::base58::base58_encode,
    encoding::base58::base58_encode_32, encoding::bech32::encode_p2wpkh_address,
    search::xoroshiro::generate_base64_nonce,
};
// === Arithmetic primitive bisect (slots 31-40) ===
// The composed primitives above all reduce to the same root cause: any
// integer op that lowers to `mul.hi.u64` (multi-word multiply, divide-by-
// constant via magic-multiply) returns wrong bytes on the current device.
// These slots pin down exactly which PTX op is broken so the alpha-compiler
// regression can be reported against a one-line repro.
//
// Pattern: each `check_arith_*` baselines the expected value via a `const`
// evaluated by the *host* rustc (correct, well-tested code), then runs the
// same expression at runtime with both operands hidden behind `black_box`
// so the GPU codegen can't constant-fold. Mismatch on GPU + match on CPU =
// codegen bug isolated to that op.

const ARITH_U32_A: u32 = 0xDEADBEEF;
const ARITH_U32_B: u32 = 0x12345678;
const ARITH_U64_A: u64 = 0xDEADBEEFCAFEBABE;
const ARITH_U64_B: u64 = 0x123456789ABCDEF0;
const ARITH_U128_A: u128 = ((ARITH_U64_A as u128) << 64) | (ARITH_U64_B as u128);
const ARITH_U128_B: u128 = ((ARITH_U64_B as u128) << 64) | (ARITH_U64_A as u128);

pub fn check_arith_u32_div_var() -> u32 {
    // Two black-boxed operands — forces `div.u32` PTX op (no magic-multiply
    // folding, since the divisor isn't a known constant).
    const EXPECTED: u32 = ARITH_U32_A / 58;
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(58u32);
    (a / b == EXPECTED) as u32
}

pub fn check_arith_u32_div_const() -> u32 {
    // Variable dividend, constant divisor — rustc lowers `x / 58` to
    // `mul.hi.u32` (or `mul.wide.u32` + shift) magic-multiply. Same path
    // base58_encode_32 uses.
    const EXPECTED: u32 = ARITH_U32_A / 58;
    let a = core::hint::black_box(ARITH_U32_A);
    (a / 58 == EXPECTED) as u32
}

pub fn check_arith_u64_div_var() -> u32 {
    // Forces `div.u64` PTX op.
    const EXPECTED: u64 = ARITH_U64_A / 58;
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(58u64);
    (a / b == EXPECTED) as u32
}

pub fn check_arith_u64_div_const() -> u32 {
    // Variable dividend, constant divisor — rustc lowers `x / 58` to
    // `mul.hi.u64` (the smoking-gun op). This is THE path base58_encode_32
    // takes for its divide-by-58 reduction loop.
    const EXPECTED: u64 = ARITH_U64_A / 58;
    let a = core::hint::black_box(ARITH_U64_A);
    (a / 58 == EXPECTED) as u32
}

pub fn check_arith_u32_rem_var() -> u32 {
    // Forces `rem.u32`.
    const EXPECTED: u32 = ARITH_U32_A % 58;
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(58u32);
    (a % b == EXPECTED) as u32
}

pub fn check_arith_u64_rem_var() -> u32 {
    // Forces `rem.u64`.
    const EXPECTED: u64 = ARITH_U64_A % 58;
    let a = core::hint::black_box(ARITH_U64_A);
    let b = core::hint::black_box(58u64);
    (a % b == EXPECTED) as u32
}

pub fn check_arith_u32_mul_lo() -> u32 {
    // Forces `mul.lo.s32` / `mul.lo.u32` (low 32 bits of u32 × u32).
    const EXPECTED: u32 = ARITH_U32_A.wrapping_mul(ARITH_U32_B);
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(ARITH_U32_B);
    (a.wrapping_mul(b) == EXPECTED) as u32
}

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

pub fn check_arith_u128_mul() -> u32 {
    // Full u128 wrapping multiply. Lowers to a sequence of `mul.lo.s64` +
    // `mul.hi.u64` + `mad.lo.s64`. Exercises the carry chain rustc emits
    // for >64-bit arithmetic.
    const EXPECTED: u128 = ARITH_U128_A.wrapping_mul(ARITH_U128_B);
    let a = core::hint::black_box(ARITH_U128_A);
    let b = core::hint::black_box(ARITH_U128_B);
    (a.wrapping_mul(b) == EXPECTED) as u32
}

// === Composed-primitive sub-bisects (slots 41-45) ===

// 25-byte test vector that lives entirely in the divide-by-58 loop with no
// leading-zero pad. Exercises `base58_encode` (variable length) as opposed
// to `base58_encode_32` (fixed) already covered by slot 3.
const BASE58_VAR_INPUT: [u8; 25] = [
    0x0A, 0xF7, 0x64, 0xC1, 0xB6, 0x13, 0x3A, 0x3A, 0x0A, 0xBD, 0x7E, 0xF9, 0xC8, 0x53, 0x79, 0x1B,
    0x68, 0x7C, 0xE1, 0xE2, 0x35, 0xF9, 0xDC, 0x84, 0x66,
];
const BASE58_VAR_EXPECTED: &[u8] = b"5Qw8TAab98QrQmymczzxwkZzacMDL4MeEH";

// Bitcoin Genesis P2PKH (mainnet) — one leading 0x00 forces the
// `num_leading_zeros` pad branch to emit a single '1' before the encoded
// numeric tail.
const BASE58_LEADZERO_INPUT: [u8; 25] = [
    0x00, 0x62, 0xE9, 0x07, 0xB1, 0x5C, 0xBF, 0x27, 0xD5, 0x42, 0x53, 0x99, 0xEB, 0xF6, 0xF0, 0xFB,
    0x50, 0xEB, 0xB8, 0x8F, 0x18, 0xC2, 0x9B, 0x7D, 0x93,
];
const BASE58_LEADZERO_EXPECTED: &[u8] = b"1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa";

// 32 all-zero bytes → all-leading-zero pad with no divide loop iterations.
// If this PASSes but slot 3 FAILs, the divide-by-58 codegen is to blame;
// if it FAILs the leading-zero pad logic itself is broken.
const BASE58_ALLZERO_EXPECTED: &[u8] = b"11111111111111111111111111111111";

pub fn check_base58_var_len() -> u32 {
    let mut out = [0u8; 64];
    let n = base58_encode(&BASE58_VAR_INPUT, &mut out);
    if n != BASE58_VAR_EXPECTED.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != BASE58_VAR_EXPECTED[i] {
            return 0;
        }
        i += 1;
    }
    1
}

pub fn check_base58_var_len_leading_zero() -> u32 {
    let mut out = [0u8; 64];
    let n = base58_encode(&BASE58_LEADZERO_INPUT, &mut out);
    if n != BASE58_LEADZERO_EXPECTED.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != BASE58_LEADZERO_EXPECTED[i] {
            return 0;
        }
        i += 1;
    }
    1
}

pub fn check_base58_all_zeros() -> u32 {
    let input = core::hint::black_box([0u8; 32]);
    let mut out = [0u8; 64];
    let n = base58_encode_32(&input, &mut out);
    if n != BASE58_ALLZERO_EXPECTED.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != BASE58_ALLZERO_EXPECTED[i] {
            return 0;
        }
        i += 1;
    }
    1
}

// Captured by running `generate_base64_nonce(0, 12345, &mut [0u8; 21])` on
// the host (see /tmp/probe). Same (thread_idx, rng_seed) the shallenge
// pipeline uses, so this slot directly answers "is the nonce wrong, and is
// that why shallenge_hash fails?".
const XOROSHIRO_NONCE_EXPECTED: [u8; 21] = [
    0x61, 0x63, 0x65, 0x43, 0x48, 0x73, 0x71, 0x46, 0x36, 0x67, 0x31, 0x33, 0x5a, 0x65, 0x32, 0x6e,
    0x47, 0x53, 0x4a, 0x67, 0x6d,
];

pub fn check_xoroshiro_base64_nonce() -> u32 {
    let mut nonce = [0u8; 21];
    generate_base64_nonce(0, 12345, &mut nonce);
    (nonce == XOROSHIRO_NONCE_EXPECTED) as u32
}

// p2wpkh KAT lifted from bech32::test::should_encode_p2wpkh_correctly.
const BECH32_P2WPKH_HASH: [u8; 20] = [
    0x46, 0x04, 0x7c, 0x8a, 0x3d, 0x8e, 0xdb, 0x13, 0x4c, 0x3f, 0x1a, 0x3e, 0x7d, 0x65, 0xb0, 0xfd,
    0x74, 0x21, 0xf1, 0x27,
];
const BECH32_P2WPKH_EXPECTED: &[u8] = b"bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m";

pub fn check_bech32_p2wpkh() -> u32 {
    let mut out = [0u8; 64];
    let n = encode_p2wpkh_address(&BECH32_P2WPKH_HASH, true, &mut out);
    if n != BECH32_P2WPKH_EXPECTED.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != BECH32_P2WPKH_EXPECTED[i] {
            return 0;
        }
        i += 1;
    }
    1
}

// === Tier-2 arithmetic bisect (slots 46-56) ===
// Targets the PTX idioms heavily used by dalek/k256 that the tier-1 net
// (slots 31-40) doesn't directly exercise: carry-chain plumbing, both-lane
// extraction from a single widening mul, fused mul-add, 32×32→64, the
// subtle::Choice mask-blend pattern, and variable shifts.

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

pub fn check_arith_mul_wide_u32() -> u32 {
    // Both operands start as u32 then widen to u64 for the mul — rustc may
    // emit `mul.wide.u32` (one PTX op, distinct from `mul.lo.u64`). k256's
    // 32-bit big-int paths take exactly this shape.
    const EXPECTED: u64 = (ARITH_U32_A as u64) * (ARITH_U32_B as u64);
    let a = core::hint::black_box(ARITH_U32_A);
    let b = core::hint::black_box(ARITH_U32_B);
    ((a as u64) * (b as u64) == EXPECTED) as u32
}

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

pub fn check_arith_var_shr_u64() -> u32 {
    // Runtime shift amount — emits `shr.b64 %rd, %rd, %r` (variable form),
    // distinct from constant-amount shifts which can be folded. Montgomery
    // reductions in k256 do variable shifts during scalar splitting.
    const EXPECTED: u64 = ARITH_U64_A >> 13;
    let a = core::hint::black_box(ARITH_U64_A);
    let n = core::hint::black_box(13u32);
    (a >> n == EXPECTED) as u32
}

pub fn check_arith_var_shl_u64() -> u32 {
    // Same as var_shr but the other direction (`shl.b64`).
    const EXPECTED: u64 = ARITH_U64_A << 13;
    let a = core::hint::black_box(ARITH_U64_A);
    let n = core::hint::black_box(13u32);
    (a << n == EXPECTED) as u32
}

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

pub fn check_arith_blackbox_identity_u32() -> u32 {
    // u32 variant — same probe at half the width in case the bug is
    // type-specific.
    let v: u32 = 0xDEADBEEF;
    (core::hint::black_box(v) == v) as u32
}

pub fn check_base58_div_by_58() -> u32 {
    // Exact divmod-by-58 pattern from base58_encode_32's digit-extraction
    // loop (logic/src/base58.rs:73-82), in isolation. The constant-divisor
    // `/ 58^k` and `% 58` lowerings emit `mul.hi.u64` magic-multiply ops
    // in PTX — the smoking-gun op from earlier inspection. No alphabet
    // lookup, no output[] writes, no leading-zero pad, no dynamic loops —
    // just the arithmetic that produces digit values in [0, 58).
    //
    // If this FAILs on GPU while the surrounding non-arithmetic code (sha2,
    // ripemd, etc.) PASSes, slot 41's `Invalid __global__ read` cascade
    // is downstream of broken div/mod codegen (corrupted digit values →
    // garbage byte writes → corrupted state → wild address used as an
    // alphabet index), not an independent OOB bug.
    const LIMB: u64 = 0x0123_4567_89AB_CDEF;
    const EXPECTED: [u8; 5] = [
        ((LIMB / 1) % 58) as u8,
        ((LIMB / 58) % 58) as u8,
        ((LIMB / (58 * 58)) % 58) as u8,
        ((LIMB / (58 * 58 * 58)) % 58) as u8,
        ((LIMB / (58 * 58 * 58 * 58)) % 58) as u8,
    ];

    let limb = core::hint::black_box(LIMB);
    let got: [u8; 5] = [
        ((limb / 1) % 58) as u8,
        ((limb / 58) % 58) as u8,
        ((limb / (58 * 58)) % 58) as u8,
        ((limb / (58 * 58 * 58)) % 58) as u8,
        ((limb / (58 * 58 * 58 * 58)) % 58) as u8,
    ];

    (got == EXPECTED) as u32
}

pub fn check_iter_static_table_lookup() -> u32 {
    // Simplest possible probe for `TABLE[byte as usize]`: a single dynamic
    // index into a small static byte slice. No iterator, no &mut, no slice
    // projection — pure indexed read from a `&'static [u8; N]` plus an
    // equality check.
    const TABLE: &[u8; 4] = b"ABCD";
    let idx = core::hint::black_box(0usize);
    (TABLE[idx] == b'A') as u32
}

pub fn check_iter_mut_slice_partial() -> u32 {
    // `for val in &mut buf[..n]` over a partial slice of a stack-resident
    // fixed-size array, writing a constant. Isolates the IterMut codegen
    // from any table lookup. If this FAILs, the iter_mut over a sliced
    // `&mut [T; N]` is the broken op.
    let mut buf = [0u8; 8];
    let n = core::hint::black_box(4usize);
    for val in &mut buf[..n] {
        *val = 0xAA;
    }
    (buf[0] == 0xAA && buf[3] == 0xAA && buf[4] == 0 && buf[7] == 0) as u32
}

pub fn check_iter_mut_alphabet_lookup() -> u32 {
    // Combined: `for val in &mut buf[..n] { *val = TABLE[*val as usize]; }`
    // — the exact final-stage pattern in base58_encode_32 that runs even
    // when the divide loop is dead (slot 43's failure case). Mirror of
    // base58.rs:99-101.
    const TABLE: &[u8; 4] = b"ABCD";
    let mut buf = [0u8; 8];
    let n = core::hint::black_box(4usize);
    for val in &mut buf[..n] {
        *val = TABLE[*val as usize];
    }
    (buf[0] == b'A' && buf[3] == b'A' && buf[4] == 0 && buf[7] == 0) as u32
}

pub fn check_iter_static_slice_lookup() -> u32 {
    // Counterpart to slot 60. Identical shape — single dynamic index into
    // a small static byte table — but typed as `&'static [u8]` (slice)
    // instead of `&'static [u8; 4]` (array reference). Slot 44 already
    // hinted that slice-typed alphabets don't crash where array-ref-typed
    // ones do (compare xoroshiro `&[u8]` → FAIL no crash, vs base58/
    // bech32 `&[u8; N]` → CRASH). This slot makes the discriminator a
    // controlled one-variable test: if 60 CRASHes and 63 PASSes, the
    // backend mishandles array-ref static indexing specifically.
    const TABLE: &[u8] = b"ABCD";
    let idx = core::hint::black_box(0usize);
    (TABLE[idx] == b'A') as u32
}

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

pub fn check_base58_limb_divrem() -> u32 {
    // The exact base58_encode_32 inner-loop shape: a u32 limb loaded
    // from a stack array, shifted into the high half, added to a u64
    // carry, then div/mod by NEXT_LIMB_DIVISOR. Slot 64 covers div by
    // 58^5 from a clean u64 source; this slot covers the case where
    // the multiplicand goes into `mul.hi.u64` after being reconstructed
    // via `shl + add`. The discriminator is the operand path, not the
    // divisor.
    //
    // Runtime index defeats mem2reg so `limbs[]` actually lives in
    // local memory and the read materialises as ld.local.b32. Even if
    // the optimizer tracks the value across the local store, the
    // multiplicand `%dividend` still comes from `add.s64(carry, shl.b64(limb,
    // 32))` — that's the suspect shape.
    const D: u64 = 58_u64.pow(5);
    let mut limbs = [0u32; 8];
    let write_idx = core::hint::black_box(3usize) & 7;
    let limb_val: u32 = core::hint::black_box(0x089A_23FF_u32);
    limbs[write_idx] = limb_val;

    let carry: u64 = core::hint::black_box(0xDEAD_BEEF_u64);
    let dividend = carry.wrapping_add((limbs[write_idx] as u64) << 32);

    // Const-eval baseline computed on the host rustc.
    const EXPECTED_DIVIDEND: u64 = 0xDEAD_BEEF_u64.wrapping_add((0x089A_23FF_u64) << 32);
    const EXPECTED_Q: u64 = EXPECTED_DIVIDEND / D;
    const EXPECTED_R: u64 = EXPECTED_DIVIDEND % D;

    (dividend == EXPECTED_DIVIDEND && dividend / D == EXPECTED_Q && dividend % D == EXPECTED_R)
        as u32
}

pub fn check_dynamic_index_write() -> u32 {
    // base58_encode_32's dynamic-growth pattern in isolation:
    //   while remaining_carry > 0 && limb_count < N {
    //       limbs[limb_count] = (remaining_carry % D) as u32;
    //       remaining_carry /= D;
    //       limb_count += 1;
    //   }
    // Slot 43 ([0u8; 32] input) PASSes because this loop never runs
    // with non-zero values; slot 3 (real input) FAILs and exercises
    // it heavily. Slot 60-62 cover runtime-index *reads*; this one is
    // about *writes* where the index variable mutates across loop
    // iterations.
    //
    // Verifies all 10 slots of the resulting array against the
    // host-CPU baseline computed under `const`.
    const D: u64 = 58_u64.pow(5);
    let mut limbs = [0u32; 10];
    let mut limb_count: usize = 0;
    let mut remaining_carry = core::hint::black_box(0xDEAD_BEEF_CAFE_BABE_u64);

    while remaining_carry > 0 && limb_count < 10 {
        limbs[limb_count] = (remaining_carry % D) as u32;
        remaining_carry /= D;
        limb_count += 1;
    }

    // Host-side const-eval of the same loop. Split into two parallel
    // const fns (one returning the array, one returning the count)
    // because cuda-oxide v1.43 can't lower tuple constants whose fields
    // are arrays — same limitation that forced the parallel-primitive
    // arrays layout in slot 64.
    const fn run_growth_limbs() -> [u32; 10] {
        let mut out = [0u32; 10];
        let mut count = 0usize;
        let mut c = 0xDEAD_BEEF_CAFE_BABE_u64;
        while c > 0 && count < 10 {
            out[count] = (c % D) as u32;
            c /= D;
            count += 1;
        }
        out
    }
    const fn run_growth_count() -> usize {
        let mut count = 0usize;
        let mut c = 0xDEAD_BEEF_CAFE_BABE_u64;
        while c > 0 && count < 10 {
            c /= D;
            count += 1;
        }
        count
    }
    const EXPECTED_LIMBS: [u32; 10] = run_growth_limbs();
    const EXPECTED_COUNT: usize = run_growth_count();

    if limb_count != EXPECTED_COUNT {
        return 0;
    }
    let mut i = 0;
    while i < 10 {
        if limbs[i] != EXPECTED_LIMBS[i] {
            return 0;
        }
        i += 1;
    }
    1
}

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

// Slot 69: base58 Phase A inner-mutate phase in isolation.
//
// `base58_encode_32` runs an 8-iter outer loop. Each iter does two phases:
//   Phase A — `for i in 0..limb_count { rc += (limbs[i] << 32); limbs[i] = (rc % D) as u32; rc /= D; }`
//   Phase B — `if rc > 0 && limb_count < 10 { limbs[limb_count] = ...; limb_count += 1; }` (×2)
// Slot 67 covered Phase B. Slot 43 (all-zero input) PASSes because Phase A
// never runs (limb_count stays 0). Slot 41 (non-zero input) FAILs and the
// only path it touches that slot 43 doesn't is Phase A. This slot runs
// Phase A standalone with limb_count=1 and a non-zero limb so the loop
// executes exactly one iteration of read-shift-add-divrem-writeback.
pub fn check_base58_inner_mutate_phase() -> u32 {
    const D: u64 = 58_u64.pow(5);
    const LIMB0_IN: u32 = 0x1234_5678;
    const CHUNK_IN: u32 = 0x89AB_CDEF;
    // Const-eval baseline (same loop body, fully evaluated at compile time).
    const fn expected_limb0() -> u32 {
        let rc: u64 = (CHUNK_IN as u64) + ((LIMB0_IN as u64) << 32);
        (rc % D) as u32
    }
    const fn expected_rc() -> u64 {
        let rc: u64 = (CHUNK_IN as u64) + ((LIMB0_IN as u64) << 32);
        rc / D
    }
    const E_LIMB0: u32 = expected_limb0();
    const E_RC: u64 = expected_rc();

    let mut limbs = [0u32; 10];
    limbs[0] = core::hint::black_box(LIMB0_IN);
    let limb_count: usize = core::hint::black_box(1);

    let chunk: u32 = core::hint::black_box(CHUNK_IN);
    let mut remaining_carry: u64 = chunk as u64;
    for i in 0..limb_count {
        remaining_carry += (limbs[i] as u64) << 32;
        limbs[i] = (remaining_carry % D) as u32;
        remaining_carry /= D;
    }

    (limbs[0] == E_LIMB0 && remaining_carry == E_RC) as u32
}

// Slot 70: curve25519-dalek `clamp_integer` in isolation. The smallest
// possible dalek call — pure bit-mask on bytes 0 and 31, no field math,
// no scalar repr conversion. If THIS fails, the bug is in the dalek
// API-entry plumbing itself, not in any arithmetic.
//
// Clamp definition (RFC 7748 / dalek):
//   byte[0]  &= 0xF8        (clear bits 0,1,2)
//   byte[31] &= 0x7F        (clear bit 7)
//   byte[31] |= 0x40        (set bit 6)
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
pub fn check_dalek_scalar_round_trip_one() -> u32 {
    let mut input = [0u8; 32];
    input[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(input);
    let bytes = scalar.to_bytes();
    (bytes == input) as u32
}

// Slot 72: `EdwardsPoint::mul_base(scalar=1).compress()` must equal the
// well-known ed25519 basepoint encoding (RFC 8032). Exercises the full
// fixed-base scalar-mult path with the smallest non-trivial scalar.
//
// If slots 70/71 PASS and 72 FAILs, the bug is in mul_base / EdwardsPoint
// ops or in `compress()` (field inversion), not in the scalar plumbing.
const ED25519_BASEPOINT_COMPRESSED: [u8; 32] = [
    0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
    0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
];

pub fn check_dalek_mul_base_scalar_one() -> u32 {
    let mut scalar_bytes = [0u8; 32];
    scalar_bytes[0] = 1;
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(scalar_bytes);
    let point = curve25519_dalek::EdwardsPoint::mul_base(&scalar);
    let compressed = point.compress().to_bytes();
    (compressed == ED25519_BASEPOINT_COMPRESSED) as u32
}

// Slot 73: `SecretKey::from_bytes` for the smallest valid scalar (=1).
// Tests just the validation/wrap step (range check + GenericArray copy).
// k256 scalars are big-endian, so 1 = [0; 31] ++ [0x01].
//
// Wrapped in ManuallyDrop because SecretKey zeroizes on Drop and
// cuda-oxide does not yet emit device-side drop_in_place (same pattern
// as logic/src/secp256k1.rs).
pub fn check_k256_secret_from_bytes_one() -> u32 {
    use core::mem::ManuallyDrop;
    use k256::SecretKey;
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 1;
    let result = SecretKey::from_bytes((&priv_bytes).into());
    match result {
        Ok(sk) => {
            let _sk = ManuallyDrop::new(sk);
            1
        }
        Err(_) => 0,
    }
}

// Slot 74: full k256 derive for scalar=1. Compressed public key must
// equal the well-known secp256k1 generator G.
const SECP256K1_GENERATOR_COMPRESSED: [u8; 33] = [
    0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
    0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17,
    0x98,
];

pub fn check_k256_derive_scalar_one() -> u32 {
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 1;
    let pub_key = secp256k1_derive_public_key(&priv_bytes);
    (pub_key == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 75: full k256 derive for scalar=2. Compressed public key must
// equal 2G (one more doubling beyond slot 74). A 74-PASS / 75-FAIL split
// pinpoints the doubling formula; a 74-FAIL / 75-FAIL means scalar mult
// is broken even for the trivial-scalar case.
const SECP256K1_TWO_G_COMPRESSED: [u8; 33] = [
    0x02, 0xC6, 0x04, 0x7F, 0x94, 0x41, 0xED, 0x7D, 0x6D, 0x30, 0x45, 0x40, 0x6E, 0x95, 0xC0, 0x7C,
    0xD8, 0x5C, 0x77, 0x8E, 0x4B, 0x8C, 0xEF, 0x3C, 0xA7, 0xAB, 0xAC, 0x09, 0xB9, 0x5C, 0x70, 0x9E,
    0xE5,
];

pub fn check_k256_derive_scalar_two() -> u32 {
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 2;
    let pub_key = secp256k1_derive_public_key(&priv_bytes);
    (pub_key == SECP256K1_TWO_G_COMPRESSED) as u32
}

// Slot 76: bare `&'static [u64; 5]` runtime-indexed read. The simplest
// possible test of the "element-width > 1 byte breaks &'static reads"
// hypothesis. No struct wrapper, no arithmetic on the result.
static STATIC_U64_TABLE: [u64; 5] = [
    0x0123_4567_89AB_CDEF,
    0xFEDC_BA98_7654_3210,
    0x1111_2222_3333_4444,
    0xAAAA_BBBB_CCCC_DDDD,
    0xDEAD_BEEF_CAFE_BABE,
];

pub fn check_static_u64_array_lookup() -> u32 {
    let idx = core::hint::black_box(3usize);
    let val = STATIC_U64_TABLE[idx];
    (val == 0xAAAA_BBBB_CCCC_DDDD) as u32
}

// Slot 77: same but wrapped in a single-field tuple struct — matches
// dalek's `Scalar52(pub(crate) [u64; 5])` newtype shape. If 76 PASSes
// and 77 FAILs, the bug is specifically in field projection through a
// newtype, not in the underlying array.
#[repr(transparent)]
pub struct U64Wrap5(pub [u64; 5]);

static STATIC_U64_WRAPPED: U64Wrap5 = U64Wrap5([
    0x0123_4567_89AB_CDEF,
    0xFEDC_BA98_7654_3210,
    0x1111_2222_3333_4444,
    0xAAAA_BBBB_CCCC_DDDD,
    0xDEAD_BEEF_CAFE_BABE,
]);

pub fn check_static_struct_wrapped_u64_lookup() -> u32 {
    let idx = core::hint::black_box(3usize);
    let val = STATIC_U64_WRAPPED.0[idx];
    (val == 0xAAAA_BBBB_CCCC_DDDD) as u32
}

// Slot 78: encode the secp256k1 generator point directly — no scalar mult,
// no Lazy<> table touch. Tests the projective→affine + to_encoded_point
// chain in isolation. ProjectivePoint::GENERATOR has z=1, so the affine
// conversion's field inversion is trivial; this primarily exercises the
// FieldElement→bytes serialization + parity-bit pack.
pub fn check_k256_encode_generator() -> u32 {
    use k256::ProjectivePoint;
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    let g = ProjectivePoint::GENERATOR;
    let affine = g.to_affine();
    let encoded = affine.to_encoded_point(true);
    let bytes = encoded.as_bytes();
    if bytes.len() != 33 {
        return 0;
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(bytes);
    (out == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 79: `ProjectivePoint::double()` on the generator + encode. One
// doubling = one field-mul-heavy operation that produces a projective
// point with z != 1, so the subsequent `to_affine()` requires a real
// field inversion. 78 PASS + 79 FAIL = doubling formula or non-trivial
// field inversion broken (5-wide variant of Bug C suspect).
pub fn check_k256_double_generator() -> u32 {
    use k256::ProjectivePoint;
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    let g2 = ProjectivePoint::GENERATOR.double();
    let affine = g2.to_affine();
    let encoded = affine.to_encoded_point(true);
    let bytes = encoded.as_bytes();
    if bytes.len() != 33 {
        return 0;
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(bytes);
    (out == SECP256K1_TWO_G_COMPRESSED) as u32
}

// Slot 80: k256 `Scalar::ONE` round-trip via the PrimeField trait. Mirror
// of slot 71 for k256's Scalar type. k256's Scalar wraps a `U256` from
// crypto-bigint (different layout than dalek's `Scalar52([u64; 5])`),
// so this distinguishes Bug A (dalek-specific newtype shape) from a
// broader Bug A' (any static-resident scalar repr).
pub fn check_k256_scalar_one_round_trip() -> u32 {
    use k256::Scalar;
    use k256::elliptic_curve::PrimeField;
    let s = Scalar::ONE;
    let repr = s.to_repr();
    let s2_opt = Scalar::from_repr(repr);
    let recovered: bool = s2_opt.is_some().into();
    if !recovered {
        return 0;
    }
    let s2 = s2_opt.unwrap();
    (s2 == s) as u32
}

// Slot 81: `u128 >> 52` immediate right shift, matching the exact shape
// inside dalek's `montgomery_reduce::part1`:
//   ((sum + m(p, constants::L[0])) >> 52, p)
// LLVM lowers u128 immediate shifts to multi-step 64-bit shift sequences.
// Slot 65 (i128 add chain) is fixed but doesn't cover this shape. Slot
// 55/56 cover u64 var shifts, not u128 immediate shifts.
pub fn check_arith_u128_imm_shr_52() -> u32 {
    const SUM: u128 = 0xFEDC_BA98_7654_3210_0123_4567_89AB_CDEF;
    const EXPECTED: u128 = SUM >> 52;
    let sum = core::hint::black_box(SUM);
    let shifted = sum >> 52;
    (shifted == EXPECTED) as u32
}

// Slot 82: depth-4 newtype nesting on `&'static` data. k256's `Scalar::ONE`
// is a `pub const Scalar = Self(U256::ONE)` where:
//   Scalar(U256)
//     U256 = Uint<4> { limbs: [Limb; 4] }
//       Limb(u64)
// So accessing the inner u64 requires `scalar.0.limbs[i].0` — 4 levels of
// field projection. Slot 77 tested depth-2 (`Wrap([u64; 5])` + index).
// If THIS fails, the bug is GEP through nested newtypes, not array reads.
#[repr(transparent)]
pub struct ProbeLimb(pub u64);

#[repr(C)]
pub struct ProbeUint4 {
    pub limbs: [ProbeLimb; 4],
}

#[repr(transparent)]
pub struct ProbeScalar(pub ProbeUint4);

static NESTED_ONE_PROBE: ProbeScalar = ProbeScalar(ProbeUint4 {
    limbs: [
        ProbeLimb(0x1111_2222_3333_4444),
        ProbeLimb(0x5555_6666_7777_8888),
        ProbeLimb(0x9999_AAAA_BBBB_CCCC),
        ProbeLimb(0xDDDD_EEEE_FFFF_0000),
    ],
});

pub fn check_static_depth4_newtype_nesting() -> u32 {
    let idx = core::hint::black_box(2usize);
    let v = NESTED_ONE_PROBE.0.limbs[idx].0;
    (v == 0x9999_AAAA_BBBB_CCCC) as u32
}

// Slot 83: reverse range iterator `(0..N).rev()` writing into a stack
// array. The only loop shape inside base58_encode_32's digit-extraction
// phase that isn't covered by an existing isolated slot:
//   for idx in (0..limb_count).rev() {
//       let output_offset = idx * DIGITS_PER_LIMB;
//       output[output_offset + i] = ...;
//   }
pub fn check_reverse_range_write() -> u32 {
    let limb_count: usize = core::hint::black_box(3);
    let mut out = [0u32; 10];
    for idx in (0..limb_count).rev() {
        out[idx] = (idx as u32) * 100;
    }
    const fn expected() -> [u32; 10] {
        let mut e = [0u32; 10];
        let mut idx = 3usize;
        while idx > 0 {
            idx -= 1;
            e[idx] = (idx as u32) * 100;
        }
        e
    }
    const EXPECTED: [u32; 10] = expected();
    (out == EXPECTED) as u32
}

// === Slots 84-87: ladder bisect of slot 71's call chain ===
//
// Slot 71's `Scalar::from_bytes_mod_order(x).to_bytes()` for x=1 expands
// inside dalek to:
//   1. Scalar52::from_bytes(&bytes)            — byte→limb unpack
//   2. Scalar52::mul_internal(x, R)            — 5×5 widening mul matrix
//   3. Scalar52::montgomery_reduce(xR)         — u128 chain + L reads
//   4. result.as_bytes()                       — limb→byte pack
//
// dalek's `backend` module is `pub(crate)`, so we can't reach Scalar52,
// mul_internal, montgomery_reduce, or constants::L/R from outside. To
// ladder-bisect we copy the minimum needed from dalek into this module
// verbatim — same Rust source, just compiled inside the logic crate so
// each step is callable in isolation.
//
// All names prefixed `bisect_` to make it obvious these are not the real
// dalek types, even though the code is byte-for-byte identical to
// `curve25519-dalek/src/backend/serial/u64/scalar.rs`.
mod bisect_scalar52 {
    //! Verbatim copy of the parts of `curve25519_dalek::backend::serial::u64::scalar`
    //! we need for slot 71's ladder bisect. If the GPU produces wrong
    //! results from THIS code (which is identical to what dalek runs
    //! internally), the bug is in the compiler's lowering of these
    //! specific Rust idioms, not in dalek's API surface.
    //!
    //! Source: curve25519-dalek 4.1.3.

    /// u64 * u64 = u128 multiply helper (dalek's `m`).
    #[inline(always)]
    fn m(x: u64, y: u64) -> u128 {
        (x as u128) * (y as u128)
    }

    #[derive(Copy, Clone)]
    pub struct Scalar52(pub [u64; 5]);

    /// `L` = group order of curve25519's scalar field.
    pub const L: Scalar52 = Scalar52([
        0x0002631a5cf5d3ed,
        0x000dea2f79cd6581,
        0x000000000014def9,
        0x0000000000000000,
        0x0000100000000000,
    ]);

    /// `L` * LFACTOR ≡ -1 (mod 2^52)
    pub const LFACTOR: u64 = 0x51da312547e1b;

    /// `R` = 2^260 mod L
    pub const R: Scalar52 = Scalar52([
        0x000f48bd6721e6ed,
        0x0003bab5ac67e45a,
        0x000fffffeb35e51b,
        0x000fffffffffffff,
        0x00000fffffffffff,
    ]);

    impl Scalar52 {
        pub const ZERO: Scalar52 = Scalar52([0, 0, 0, 0, 0]);

        /// Unpack 32 bytes (little-endian) into 5 52-bit limbs.
        pub fn from_bytes(bytes: &[u8; 32]) -> Scalar52 {
            let mut words = [0u64; 4];
            for i in 0..4 {
                for j in 0..8 {
                    words[i] |= (bytes[(i * 8) + j] as u64) << (j * 8);
                }
            }
            let mask = (1u64 << 52) - 1;
            let top_mask = (1u64 << 48) - 1;
            let mut s = Scalar52::ZERO;
            s.0[0] = words[0] & mask;
            s.0[1] = ((words[0] >> 52) | (words[1] << 12)) & mask;
            s.0[2] = ((words[1] >> 40) | (words[2] << 24)) & mask;
            s.0[3] = ((words[2] >> 28) | (words[3] << 36)) & mask;
            s.0[4] = (words[3] >> 16) & top_mask;
            s
        }

        /// Pack 5 52-bit limbs into 32 bytes (little-endian).
        #[allow(clippy::identity_op)]
        pub fn as_bytes(&self) -> [u8; 32] {
            let mut s = [0u8; 32];
            s[0] = (self.0[0] >> 0) as u8;
            s[1] = (self.0[0] >> 8) as u8;
            s[2] = (self.0[0] >> 16) as u8;
            s[3] = (self.0[0] >> 24) as u8;
            s[4] = (self.0[0] >> 32) as u8;
            s[5] = (self.0[0] >> 40) as u8;
            s[6] = ((self.0[0] >> 48) | (self.0[1] << 4)) as u8;
            s[7] = (self.0[1] >> 4) as u8;
            s[8] = (self.0[1] >> 12) as u8;
            s[9] = (self.0[1] >> 20) as u8;
            s[10] = (self.0[1] >> 28) as u8;
            s[11] = (self.0[1] >> 36) as u8;
            s[12] = (self.0[1] >> 44) as u8;
            s[13] = (self.0[2] >> 0) as u8;
            s[14] = (self.0[2] >> 8) as u8;
            s[15] = (self.0[2] >> 16) as u8;
            s[16] = (self.0[2] >> 24) as u8;
            s[17] = (self.0[2] >> 32) as u8;
            s[18] = (self.0[2] >> 40) as u8;
            s[19] = ((self.0[2] >> 48) | (self.0[3] << 4)) as u8;
            s[20] = (self.0[3] >> 4) as u8;
            s[21] = (self.0[3] >> 12) as u8;
            s[22] = (self.0[3] >> 20) as u8;
            s[23] = (self.0[3] >> 28) as u8;
            s[24] = (self.0[3] >> 36) as u8;
            s[25] = (self.0[3] >> 44) as u8;
            s[26] = (self.0[4] >> 0) as u8;
            s[27] = (self.0[4] >> 8) as u8;
            s[28] = (self.0[4] >> 16) as u8;
            s[29] = (self.0[4] >> 24) as u8;
            s[30] = (self.0[4] >> 32) as u8;
            s[31] = (self.0[4] >> 40) as u8;
            s
        }

        /// 5×5 widening multiply → [u128; 9]. The exact shape dalek uses.
        pub fn mul_internal(a: &Scalar52, b: &Scalar52) -> [u128; 9] {
            let mut z = [0u128; 9];
            z[0] = m(a.0[0], b.0[0]);
            z[1] = m(a.0[0], b.0[1]) + m(a.0[1], b.0[0]);
            z[2] = m(a.0[0], b.0[2]) + m(a.0[1], b.0[1]) + m(a.0[2], b.0[0]);
            z[3] = m(a.0[0], b.0[3]) + m(a.0[1], b.0[2]) + m(a.0[2], b.0[1]) + m(a.0[3], b.0[0]);
            z[4] = m(a.0[0], b.0[4])
                + m(a.0[1], b.0[3])
                + m(a.0[2], b.0[2])
                + m(a.0[3], b.0[1])
                + m(a.0[4], b.0[0]);
            z[5] = m(a.0[1], b.0[4]) + m(a.0[2], b.0[3]) + m(a.0[3], b.0[2]) + m(a.0[4], b.0[1]);
            z[6] = m(a.0[2], b.0[4]) + m(a.0[3], b.0[3]) + m(a.0[4], b.0[2]);
            z[7] = m(a.0[3], b.0[4]) + m(a.0[4], b.0[3]);
            z[8] = m(a.0[4], b.0[4]);
            z
        }

        /// Compute `a - b` (mod L). Verbatim from dalek's u64 scalar.rs.
        ///
        /// The trailing conditional-add (when the borrow underflows the
        /// low 52 bits, add L back) makes this a constant-time signed-vs-
        /// unsigned bridge. Uses a per-function `black_box` via volatile
        /// load to prevent LLVM from inserting `jns` branches.
        pub fn sub(a: &Scalar52, b: &Scalar52) -> Scalar52 {
            fn black_box(value: u64) -> u64 {
                // Same as dalek: ptr::read_volatile to defeat optimization
                unsafe { core::ptr::read_volatile(&value) }
            }
            let mut difference = Scalar52::ZERO;
            let mask = (1u64 << 52) - 1;
            let mut borrow: u64 = 0;
            for i in 0..5 {
                borrow = a.0[i].wrapping_sub(b.0[i] + (borrow >> 63));
                difference.0[i] = borrow & mask;
            }
            let underflow_mask = ((borrow >> 63) ^ 1).wrapping_sub(1);
            let mut carry: u64 = 0;
            for i in 0..5 {
                carry = (carry >> 52) + difference.0[i] + (L.0[i] & black_box(underflow_mask));
                difference.0[i] = carry & mask;
            }
            difference
        }

        /// Same as `montgomery_reduce` but stops before the final
        /// `Scalar52::sub(result, L)` call. Used by slot 85 — preserves
        /// the test point from the v1.46 run where 85 was confirmed to
        /// PASS without sub. Keeping this separate lets us directly
        /// compare "reduce without sub" (slot 85) vs "reduce with sub"
        /// (slot 90) results on each subsequent vast run.
        pub fn montgomery_reduce_no_sub(limbs: &[u128; 9]) -> Scalar52 {
            #[inline(always)]
            fn part1(sum: u128) -> (u128, u64) {
                let p = (sum as u64).wrapping_mul(LFACTOR) & ((1u64 << 52) - 1);
                ((sum + m(p, L.0[0])) >> 52, p)
            }
            #[inline(always)]
            fn part2(sum: u128) -> (u128, u64) {
                let w = (sum as u64) & ((1u64 << 52) - 1);
                (sum >> 52, w)
            }
            let l = &L;
            let (carry, n0) = part1(limbs[0]);
            let (carry, n1) = part1(carry + limbs[1] + m(n0, l.0[1]));
            let (carry, n2) = part1(carry + limbs[2] + m(n0, l.0[2]) + m(n1, l.0[1]));
            let (carry, n3) = part1(carry + limbs[3] + m(n1, l.0[2]) + m(n2, l.0[1]));
            let (carry, n4) =
                part1(carry + limbs[4] + m(n0, l.0[4]) + m(n2, l.0[2]) + m(n3, l.0[1]));
            let (carry, r0) =
                part2(carry + limbs[5] + m(n1, l.0[4]) + m(n3, l.0[2]) + m(n4, l.0[1]));
            let (carry, r1) = part2(carry + limbs[6] + m(n2, l.0[4]) + m(n4, l.0[2]));
            let (carry, r2) = part2(carry + limbs[7] + m(n3, l.0[4]));
            let (carry, r3) = part2(carry + limbs[8] + m(n4, l.0[4]));
            let r4 = carry as u64;
            Scalar52([r0, r1, r2, r3, r4])
        }

        /// Compute `limbs / R` (mod L) — exactly dalek's montgomery_reduce
        /// including the final `Scalar52::sub(result, L)` call.
        pub fn montgomery_reduce(limbs: &[u128; 9]) -> Scalar52 {
            #[inline(always)]
            fn part1(sum: u128) -> (u128, u64) {
                let p = (sum as u64).wrapping_mul(LFACTOR) & ((1u64 << 52) - 1);
                ((sum + m(p, L.0[0])) >> 52, p)
            }
            #[inline(always)]
            fn part2(sum: u128) -> (u128, u64) {
                let w = (sum as u64) & ((1u64 << 52) - 1);
                (sum >> 52, w)
            }
            let l = &L;
            let (carry, n0) = part1(limbs[0]);
            let (carry, n1) = part1(carry + limbs[1] + m(n0, l.0[1]));
            let (carry, n2) = part1(carry + limbs[2] + m(n0, l.0[2]) + m(n1, l.0[1]));
            let (carry, n3) = part1(carry + limbs[3] + m(n1, l.0[2]) + m(n2, l.0[1]));
            let (carry, n4) =
                part1(carry + limbs[4] + m(n0, l.0[4]) + m(n2, l.0[2]) + m(n3, l.0[1]));
            let (carry, r0) =
                part2(carry + limbs[5] + m(n1, l.0[4]) + m(n3, l.0[2]) + m(n4, l.0[1]));
            let (carry, r1) = part2(carry + limbs[6] + m(n2, l.0[4]) + m(n4, l.0[2]));
            let (carry, r2) = part2(carry + limbs[7] + m(n3, l.0[4]));
            let (carry, r3) = part2(carry + limbs[8] + m(n4, l.0[4]));
            let r4 = carry as u64;
            // The full dalek implementation: result may be >= L, so
            // attempt to subtract L. This was missing from earlier
            // iterations of the port — slot 86 PASSed without it, which
            // told us mul_internal + reduce-without-sub are fine but
            // hid the fact that sub itself might be the bug.
            Scalar52::sub(&Scalar52([r0, r1, r2, r3, r4]), l)
        }
    }
}

const DALEK_ONE_LIMBS: [u64; 5] = [1, 0, 0, 0, 0];

// Slot 84 — Rung A: pure byte→limbs unpack. No arithmetic, no statics
// other than the const masks.
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

// Slot 91: focused Index/IndexMut trait dispatch probe on a tuple
// struct. Mirrors dalek's Scalar52 Index impl shape EXACTLY: tuple
// struct wrapping `[u64; 5]`, Index returns `&u64`, IndexMut returns
// `&mut u64`. If this FAILs, trait dispatch on `[i]` syntax is broken
// on the cuda-oxide alpha-NVPTX backend — explains why dalek (uses
// `a[i]`) fails while our port (uses `a.0[i]`) passes.
pub struct IdxProbe(pub [u64; 5]);

impl core::ops::Index<usize> for IdxProbe {
    type Output = u64;
    fn index(&self, i: usize) -> &u64 {
        &(self.0[i])
    }
}

impl core::ops::IndexMut<usize> for IdxProbe {
    fn index_mut(&mut self, i: usize) -> &mut u64 {
        &mut (self.0[i])
    }
}

pub fn check_index_trait_dispatch() -> u32 {
    let mut p = IdxProbe([0u64; 5]);
    let idx = core::hint::black_box(2usize);
    let val = core::hint::black_box(0xCAFE_BABE_DEAD_BEEF_u64);
    p[idx] = val;
    let read = core::hint::black_box(p[idx]);
    (read == val) as u32
}

// Slot 92: dalek `Scalar::ONE.to_bytes()` direct. Cross-crate access to
// a `pub const Scalar` followed by trivial byte copy (Scalar's internal
// rep IS the bytes; to_bytes just copies them out). No reduce, no math.
// If this FAILs, the bug is at the cross-crate const-access layer.
pub fn check_dalek_scalar_one_to_bytes_direct() -> u32 {
    let s = core::hint::black_box(curve25519_dalek::Scalar::ONE);
    let bytes = s.to_bytes();
    let mut expected = [0u8; 32];
    expected[0] = 1;
    (bytes == expected) as u32
}

// Slot 93: k256 `AffinePoint::GENERATOR.to_encoded_point(true)`. Skips
// the projective→affine conversion that slot 78 includes (no z-coord
// inversion). Tests cross-crate const access for AffinePoint::GENERATOR
// + the encoded_point serialization chain. If 93 PASSes and 78 FAILs,
// the bug in 78 is specifically in `to_affine()` (the field inversion).
pub fn check_k256_affine_generator_encode() -> u32 {
    use k256::AffinePoint;
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    let g = AffinePoint::GENERATOR;
    let encoded = g.to_encoded_point(true);
    let bytes = encoded.as_bytes();
    if bytes.len() != 33 {
        return 0;
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(bytes);
    (out == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 94: subtle::Choice u8 → bool. The most trivial subtle operation.
// Choice is a tuple struct wrapping u8 with field private. From<u8> sets
// it; Into<bool> reads it via debug_assert + comparison.
pub fn check_subtle_choice_u8_into_bool() -> u32 {
    use k256::elliptic_curve::subtle::Choice;
    let c0 = Choice::from(core::hint::black_box(0u8));
    let c1 = Choice::from(core::hint::black_box(1u8));
    let b0: bool = c0.into();
    let b1: bool = c1.into();
    (!b0 && b1) as u32
}

// Slot 95: subtle::ConditionallySelectable on u64. The mechanism k256's
// `AffinePoint::to_encoded_point` uses to pick between the identity
// arm and the from_affine_coordinates arm.
//   conditional_select(&a, &b, Choice(0)) should return a
//   conditional_select(&a, &b, Choice(1)) should return b
// Slot 53/54 tested a HAND-ROLLED mask blend with the same conceptual
// math; this slot tests the actual subtle::ConditionallySelectable trait
// impl which the real code path uses.
pub fn check_subtle_conditional_select_u64() -> u32 {
    use k256::elliptic_curve::subtle::{Choice, ConditionallySelectable};
    let a = core::hint::black_box(0xCAFE_BABE_DEAD_BEEF_u64);
    let b = core::hint::black_box(0x1234_5678_9ABC_DEF0_u64);
    let c0 = Choice::from(core::hint::black_box(0u8));
    let c1 = Choice::from(core::hint::black_box(1u8));
    let r0 = u64::conditional_select(&a, &b, c0);
    let r1 = u64::conditional_select(&a, &b, c1);
    (r0 == a && r1 == b) as u32
}

// Slot 96: `EncodedPoint::from_affine_coordinates(&GX_bytes, &GY_bytes,
// compress=true)` with hardcoded generator-x/y. Bypasses AffinePoint's
// own `to_encoded_point` (which goes through `is_identity`+
// `conditional_select`) and tests just the EncodedPoint construction.
//
// If 96 PASSes and 93 FAILs, the bug is in `is_identity`/`conditional_
// select` (slot 95 should then also FAIL). If 96 FAILs, EncodedPoint
// construction itself is broken.
const SECP256K1_GX_BYTES: [u8; 32] = [
    0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
    0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
];
const SECP256K1_GY_BYTES: [u8; 32] = [
    0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65, 0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
    0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19, 0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8,
];

pub fn check_k256_encoded_point_from_affine_coords() -> u32 {
    use k256::EncodedPoint;
    use k256::elliptic_curve::FieldBytes;
    let x_bytes = core::hint::black_box(SECP256K1_GX_BYTES);
    let y_bytes = core::hint::black_box(SECP256K1_GY_BYTES);
    let x: &FieldBytes<k256::Secp256k1> = (&x_bytes).into();
    let y: &FieldBytes<k256::Secp256k1> = (&y_bytes).into();
    let encoded = EncodedPoint::from_affine_coordinates(x, y, true);
    let bytes = encoded.as_bytes();
    if bytes.len() != 33 {
        return 0;
    }
    let mut out = [0u8; 33];
    out.copy_from_slice(bytes);
    (out == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 97: Index/IndexMut trait dispatch with LITERAL const indices.
// Slot 91 used `black_box(idx)` → runtime index, and now PASSes. Dalek's
// Scalar52::from_bytes uses `s[0] = …; s[1] = …; …; s[4] = …` with
// const literal indices. Different IR shape — const indices typically
// fold the trait call into a direct GEP at compile time.
pub fn check_index_trait_const_indices() -> u32 {
    let mut p = IdxProbe([0u64; 5]);
    p[0] = core::hint::black_box(0x1111_1111_1111_1111_u64);
    p[1] = core::hint::black_box(0x2222_2222_2222_2222_u64);
    p[2] = core::hint::black_box(0x3333_3333_3333_3333_u64);
    p[3] = core::hint::black_box(0x4444_4444_4444_4444_u64);
    p[4] = core::hint::black_box(0x5555_5555_5555_5555_u64);
    let r0 = p[0];
    let r1 = p[1];
    let r2 = p[2];
    let r3 = p[3];
    let r4 = p[4];
    (r0 == 0x1111_1111_1111_1111
        && r1 == 0x2222_2222_2222_2222
        && r2 == 0x3333_3333_3333_3333
        && r3 == 0x4444_4444_4444_4444
        && r4 == 0x5555_5555_5555_5555) as u32
}

// Slot 98: `GenericArray<u8, U33>` basic index. GenericArray doesn't have
// a custom Index impl; it Derefs to `[T]` via:
//   `unsafe { slice::from_raw_parts(self as *const Self as *const T, N::USIZE) }`
// If that raw-ptr-cast Deref miscompiles, every GenericArray op breaks.
// k256::EncodedPoint stores its bytes in a `GenericArray<u8, EncodedSize>`.
pub fn check_generic_array_basic_index() -> u32 {
    use k256::elliptic_curve::generic_array::GenericArray;
    use k256::elliptic_curve::generic_array::typenum::U33;
    let mut ga: GenericArray<u8, U33> = GenericArray::default();
    let i0 = core::hint::black_box(0usize);
    let i32 = core::hint::black_box(32usize);
    ga[i0] = 0xAA;
    ga[i32] = 0xBB;
    let v0 = ga[i0];
    let v32 = ga[i32];
    (v0 == 0xAA && v32 == 0xBB) as u32
}

// Slot 99: `GenericArray<u8, U33>` populated via `copy_from_slice` from a
// regular byte array. This is exactly what EncodedPoint::from_affine_
// coordinates does:
//   bytes[1..33].copy_from_slice(x);
// If this FAILs, the slice-copy-into-GenericArray-slice is the bug.
pub fn check_generic_array_copy_from_slice() -> u32 {
    use k256::elliptic_curve::generic_array::GenericArray;
    use k256::elliptic_curve::generic_array::typenum::U33;
    let src: [u8; 32] = core::hint::black_box(SECP256K1_GX_BYTES);
    let mut ga: GenericArray<u8, U33> = GenericArray::default();
    ga[0] = 0x02;
    ga[1..33].copy_from_slice(&src);
    // Compare against the known compressed-generator encoding.
    let mut got = [0u8; 33];
    got.copy_from_slice(&ga[..]);
    (got == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 100: local re-impl of sec1's `from_affine_coordinates` body using
// raw `[u8; 33]` instead of `GenericArray<u8, U33>`. Same algorithm:
//   tag = 0x02/0x03 based on y[31]&1
//   bytes[0] = tag
//   bytes[1..33] = x
// If 100 PASSes and 96 still FAILs, the bug is in sec1's
// GenericArray-typed parameter handling, not the algorithm.
pub fn check_from_affine_coords_replica() -> u32 {
    let x_bytes = &SECP256K1_GX_BYTES;
    let y_bytes = &SECP256K1_GY_BYTES;
    // Compute tag: even y → 0x02, odd y → 0x03
    let last_y = core::hint::black_box(y_bytes[31]);
    let tag: u8 = if last_y & 1 == 1 { 0x03 } else { 0x02 };
    let mut bytes = [0u8; 33];
    bytes[0] = tag;
    bytes[1..33].copy_from_slice(x_bytes);
    (bytes == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 101: probes the exact `y.as_slice().last()` shape inside
// `Tag::compress_y`. Pass a `&GenericArray<u8, U32>` to a function, do
// `as_slice().last()` inside. Slot 99 tested write-side copy; this
// tests read-side slice access via Deref then `.last()`.
#[inline(never)]
fn last_via_as_slice(
    ga: &k256::elliptic_curve::generic_array::GenericArray<
        u8,
        k256::elliptic_curve::generic_array::typenum::U32,
    >,
) -> u8 {
    *ga.as_slice().last().expect("non-empty")
}

pub fn check_generic_array_as_slice_last() -> u32 {
    use k256::elliptic_curve::generic_array::GenericArray;
    use k256::elliptic_curve::generic_array::typenum::U32;
    let input = core::hint::black_box(SECP256K1_GY_BYTES);
    let ga: &GenericArray<u8, U32> = (&input).into();
    let last = last_via_as_slice(ga);
    (last == 0xB8) as u32 // SECP256K1_GY_BYTES[31]
}

// Slot 102: dalek `Scalar::from_bytes_mod_order([0; 32])` should
// round-trip to all-zeros (the canonical encoding of 0). Mirror of slot
// 71 with a different input value. If 102 PASS but 71 FAIL, the bug is
// input-dependent (only non-zero scalars). If 102 FAIL too, the bug is
// general to any Scalar::from_bytes_mod_order call.
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
pub fn check_dalek_scalar_from_bytes_wide_zero() -> u32 {
    let input = [0u8; 64];
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order_wide(&core::hint::black_box(input));
    let bytes = scalar.to_bytes();
    (bytes == [0u8; 32]) as u32
}

// Slot 104: `(&[u8; 32]).into() → &FieldBytes<Secp256k1>` then read first
// and last bytes. Tests the `From<&[u8; N]> for &GenericArray<u8, N>`
// conversion (the only GA-related path slot 98/99 didn't cover — they
// constructed via `GenericArray::default()` instead).
pub fn check_field_bytes_into_conversion() -> u32 {
    use k256::elliptic_curve::FieldBytes;
    let arr: [u8; 32] = SECP256K1_GX_BYTES;
    let arr = core::hint::black_box(arr);
    let ga: &FieldBytes<k256::Secp256k1> = (&arr).into();
    let first = ga[0];
    let last = ga[31];
    (first == 0x79 && last == 0x98) as u32
}

// Slot 105: `base58_encode_32` with minimum non-zero input: 31 leading
// zero bytes + 1 byte of value 1. Forces `limb_count == 1` after the
// outer loop (vs slot 41 which has higher limb_count). Expected output
// is 31 '1's followed by '2' = 32 chars.
pub fn check_base58_min_nonzero() -> u32 {
    let mut input = [0u8; 32];
    input[31] = 1;
    let mut out = [0u8; 64];
    let n = base58_encode_32(&input, &mut out);
    let expected: &[u8] = b"11111111111111111111111111111112";
    if n != expected.len() {
        return 0;
    }
    let mut i = 0;
    while i < n {
        if out[i] != expected[i] {
            return 0;
        }
        i += 1;
    }
    1
}

// Slot 106: Named-field struct wrapping `[u8; 32]` return-by-value test.
// This is the EXACT shape of dalek's `Scalar`:
//   pub struct Scalar { pub(crate) bytes: [u8; 32] }
// All known-passing return shapes:
//   - `clamp_integer` returns `[u8; 32]` direct (slot 70 PASS)
//   - `Scalar52::from_bytes` returns tuple-struct `Scalar52(pub [u64; 5])` (slot 84 PASS)
//   - `Scalar::ONE.to_bytes()` returns `[u8; 32]` direct (slot 92 PASS)
// All known-failing through dalek's Scalar:
//   - `Scalar::from_bytes_mod_order` returns Scalar (named-field struct) (slot 71/102/103 FAIL)
// Hypothesis: returning a named-field struct wrapping `[u8; 32]` by
// value is miscompiled. If 106 FAILs, that's the minimum Bug-71 repro.
#[repr(C)]
pub struct WrapNamed {
    pub bytes: [u8; 32],
}

#[inline(never)]
fn make_wrap_named(input: [u8; 32]) -> WrapNamed {
    let mut bytes = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        bytes[i] = input[i].wrapping_add(1);
        i += 1;
    }
    WrapNamed { bytes }
}

pub fn check_named_field_struct_return() -> u32 {
    let input = core::hint::black_box([0u8; 32]);
    let out = make_wrap_named(input);
    let mut expected = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        expected[i] = 1;
        i += 1;
    }
    (out.bytes == expected) as u32
}

// Slot 107: hand-rolled base58 of [0; 31] + [0x01] without the `seq!`
// macro. base58_encode_32 uses `seq!(I in 0..8 { ... })` which proc-macro-
// unrolls the outer loop 8 times. This replica uses a plain `for I in
// 0..8` instead, with otherwise-identical body. Same algorithm, same
// constants, just no seq! expansion.
//
// If 107 PASS but 105 FAIL, the bug is in something specific to seq!'s
// expansion (function size, code layout, etc).
// Slot 108: `<[u8]>::reverse()` on a partial sub-slice. The only
// operation in `base58_encode_32` that slot 107 (which PASSed) hand-
// rolls — slot 107 uses manual swap pairs, while the original calls
// `output[..result_len].reverse()`. If 108 FAILs, that's the Bug-41
// minimal repro.
pub fn check_slice_reverse_partial() -> u32 {
    let mut arr = [0u8; 64];
    // Populate a non-trivial prefix with a recognizable pattern
    arr[0] = 0x11;
    arr[1] = 0x22;
    arr[2] = 0x33;
    arr[3] = 0x44;
    arr[4] = 0x55;
    let result_len = core::hint::black_box(5usize);
    arr[..result_len].reverse();
    // Expected after reverse: [0x55, 0x44, 0x33, 0x22, 0x11, 0, 0, ...]
    let mut expected = [0u8; 64];
    expected[0] = 0x55;
    expected[1] = 0x44;
    expected[2] = 0x33;
    expected[3] = 0x22;
    expected[4] = 0x11;
    (arr == expected) as u32
}

// Slot 109: `Scalar::from_bytes_mod_order([0; 32]) == Scalar::ZERO`
// using dalek's `PartialEq` (which uses constant-time equality
// internally) instead of comparing the bytes output of `to_bytes`.
// Disambiguates: is the Scalar VALUE correct, or is `to_bytes` broken?
//   If 109 PASS but 102 FAIL → bug is specifically in `to_bytes`.
//   If 109 FAIL → the Scalar value from `from_bytes_mod_order` is wrong.
pub fn check_dalek_scalar_eq_zero() -> u32 {
    use curve25519_dalek::Scalar;
    let input = core::hint::black_box([0u8; 32]);
    let s = Scalar::from_bytes_mod_order(input);
    let zero = Scalar::ZERO;
    (s == zero) as u32
}

// Slot 110: `dst_ga.copy_from_slice(src_ga)` where source IS a
// `&GenericArray<u8, U32>` (not `&[u8; 32]`). Slot 99 already covered
// `&[u8; 32]` source. The function `EncodedPoint::from_affine_coordinates`
// uses `bytes[1..33].copy_from_slice(x)` where `x: &GenericArray`, so
// the source-side Deref→slice conversion happens implicitly.
pub fn check_generic_array_copy_from_ga_source() -> u32 {
    use k256::elliptic_curve::generic_array::GenericArray;
    use k256::elliptic_curve::generic_array::typenum::{U32, U33};
    let src_arr = core::hint::black_box(SECP256K1_GX_BYTES);
    let src: &GenericArray<u8, U32> = (&src_arr).into();
    let mut dst: GenericArray<u8, U33> = GenericArray::default();
    dst[0] = 0x02;
    dst[1..33].copy_from_slice(src);
    let mut got = [0u8; 33];
    got.copy_from_slice(&dst);
    (got == SECP256K1_GENERATOR_COMPRESSED) as u32
}

// Slot 111: `Scalar::ZERO == Scalar::ZERO`. Pure const-vs-const
// PartialEq, no function call producing a Scalar. If FAIL, dalek's
// PartialEq impl itself is broken; if PASS, slot 109's FAIL is
// genuinely from from_bytes_mod_order returning a non-zero value.
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
pub fn check_dalek_scalar52_from_bytes_zero() -> u32 {
    let bytes = core::hint::black_box([0u8; 32]);
    let s = bisect_scalar52::Scalar52::from_bytes(&bytes);
    (s.0 == [0u64; 5]) as u32
}

// Slot 114: `Scalar52::mul_internal(ZERO, R)` via the verbatim port. The
// 5x5 widening multiply matrix should produce all-zero u128 limbs when
// one operand is zero. Slot 86 covered ONE * R; this is the zero case.
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
pub fn check_dalek_scalar52_montgomery_reduce_zero() -> u32 {
    let widened = core::hint::black_box([0u128; 9]);
    let result = bisect_scalar52::Scalar52::montgomery_reduce(&widened);
    (result.0 == [0u64; 5]) as u32
}

// Slot 116: `Scalar52::ZERO.as_bytes() == [0; 32]` — pack of zero via
// the verbatim port. Slot 87 covered pack of ONE. Completing this
// confirms every individual reduce primitive works on zero in isolation.
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
pub fn check_dalek_reduce_pipeline_zero() -> u32 {
    use bisect_scalar52::{R, Scalar52};
    let bytes = core::hint::black_box([0u8; 32]);
    let x = Scalar52::from_bytes(&bytes);
    let x_r = Scalar52::mul_internal(&x, &R);
    let reduced = Scalar52::montgomery_reduce(&x_r);
    let out = reduced.as_bytes();
    (out == [0u8; 32]) as u32
}

pub fn check_base58_handrolled_no_seq() -> u32 {
    const D: u64 = 58_u64.pow(5);
    const DIVISORS: [u64; 5] = [1, 58, 3364, 195112, 11316496];
    const BASE58_ALPHABET: &[u8; 58] =
        b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    let mut input = [0u8; 32];
    input[31] = 1;
    let input = core::hint::black_box(input);

    // num_leading_zeros
    let mut num_leading_zeros: usize = 0;
    let mut i = 0;
    while i < 32 {
        if input[i] == 0 {
            num_leading_zeros += 1;
        } else {
            break;
        }
        i += 1;
    }

    // chunks
    let mut chunks = [0u32; 8];
    let mut c = 0;
    while c < 8 {
        chunks[c] = u32::from_be_bytes([
            input[c * 4],
            input[c * 4 + 1],
            input[c * 4 + 2],
            input[c * 4 + 3],
        ]);
        c += 1;
    }

    // outer loop — same body as base58_encode_32 but plain Rust, no seq!
    let mut limbs = [0u32; 10];
    let mut limb_count: usize = 0;
    let mut k = 0;
    while k < 8 {
        let chunk = chunks[k];
        let carry = chunk as u64;
        let mut remaining_carry = carry;

        let mut j = 0;
        while j < limb_count {
            remaining_carry += (limbs[j] as u64) << 32;
            limbs[j] = (remaining_carry % D) as u32;
            remaining_carry /= D;
            j += 1;
        }

        if remaining_carry > 0 && limb_count < 10 {
            limbs[limb_count] = (remaining_carry % D) as u32;
            remaining_carry /= D;
            limb_count += 1;
            if remaining_carry > 0 && limb_count < 10 {
                limbs[limb_count] = remaining_carry as u32;
                limb_count += 1;
            }
        }
        k += 1;
    }

    // digit extraction
    let mut output = [0u8; 64];
    let mut idx = limb_count;
    while idx > 0 {
        idx -= 1;
        let limb_value = limbs[idx] as u64;
        let output_offset = idx * 5;
        let mut di = 0;
        while di < 5 {
            output[output_offset + di] = ((limb_value / DIVISORS[di]) % 58) as u8;
            di += 1;
        }
    }

    let mut result_len = limb_count * 5;
    while result_len > 0 && output[result_len - 1] == 0 {
        result_len -= 1;
    }

    let mut z = 0;
    while z < num_leading_zeros {
        output[result_len] = 0;
        result_len += 1;
        z += 1;
    }

    let mut a = 0;
    while a < result_len {
        output[a] = BASE58_ALPHABET[output[a] as usize];
        a += 1;
    }

    // reverse output[..result_len]
    let mut lo = 0;
    let mut hi = result_len;
    while lo + 1 < hi {
        hi -= 1;
        let tmp = output[lo];
        output[lo] = output[hi];
        output[hi] = tmp;
        lo += 1;
    }

    let expected: &[u8] = b"11111111111111111111111111111112";
    if result_len != expected.len() {
        return 0;
    }
    let mut x = 0;
    while x < result_len {
        if output[x] != expected[x] {
            return 0;
        }
        x += 1;
    }
    1
}
