//! Concrete layout probes used by this mode's device self-test.
use super::*;

#[inline(never)]
pub fn check_iter_static_table_lookup() -> u32 {
    // Simplest possible probe for `TABLE[byte as usize]`: a single dynamic
    // index into a small static byte slice. No iterator, no &mut, no slice
    // projection — pure indexed read from a `&'static [u8; N]` plus an
    // equality check.
    const TABLE: &[u8; 4] = b"ABCD";
    let idx = core::hint::black_box(0usize);
    (TABLE[idx] == b'A') as u32
}

#[inline(never)]
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

#[inline(never)]
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

#[inline(never)]
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

#[inline(never)]
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

#[inline(never)]
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
#[inline(never)]
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

#[inline(never)]
pub fn check_index_trait_dispatch() -> u32 {
    let mut p = IdxProbe([0u64; 5]);
    let idx = core::hint::black_box(2usize);
    let val = core::hint::black_box(0xCAFE_BABE_DEAD_BEEF_u64);
    p[idx] = val;
    let read = core::hint::black_box(p[idx]);
    (read == val) as u32
}

#[inline(never)]
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
#[inline(never)]
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
