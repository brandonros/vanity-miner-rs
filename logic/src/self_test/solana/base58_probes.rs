//! Concrete base58 probes used by this mode's device self-test.
use super::*;

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
