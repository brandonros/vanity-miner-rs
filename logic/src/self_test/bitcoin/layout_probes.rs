//! Concrete layout probes used by this mode's device self-test.
use super::*;

#[inline(never)]
pub fn check_static_u64_array_lookup() -> u32 {
    let idx = core::hint::black_box(3usize);
    let val = STATIC_U64_TABLE[idx];
    (val == 0xAAAA_BBBB_CCCC_DDDD) as u32
}

#[inline(never)]
pub fn check_static_struct_wrapped_u64_lookup() -> u32 {
    let idx = core::hint::black_box(3usize);
    let val = STATIC_U64_WRAPPED.0[idx];
    (val == 0xAAAA_BBBB_CCCC_DDDD) as u32
}

// Slot 94: subtle::Choice u8 → bool. The most trivial subtle operation.
// Choice is a tuple struct wrapping u8 with field private. From<u8> sets
// it; Into<bool> reads it via debug_assert + comparison.
#[inline(never)]
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
#[inline(never)]
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

// Slot 97: Index/IndexMut trait dispatch with LITERAL const indices.
// Slot 91 used `black_box(idx)` → runtime index, and now PASSes. Dalek's
// Scalar52::from_bytes uses `s[0] = …; s[1] = …; …; s[4] = …` with
// const literal indices. Different IR shape — const indices typically
// fold the trait call into a direct GEP at compile time.
#[inline(never)]
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
#[inline(never)]
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
#[inline(never)]
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
#[inline(never)]
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

#[inline(never)]
pub fn check_generic_array_as_slice_last() -> u32 {
    use k256::elliptic_curve::generic_array::GenericArray;
    use k256::elliptic_curve::generic_array::typenum::U32;
    let input = core::hint::black_box(SECP256K1_GY_BYTES);
    let ga: &GenericArray<u8, U32> = (&input).into();
    let last = last_via_as_slice(ga);
    (last == 0xB8) as u32 // SECP256K1_GY_BYTES[31]
}

// Slot 104: `(&[u8; 32]).into() → &FieldBytes<Secp256k1>` then read first
// and last bytes. Tests the `From<&[u8; N]> for &GenericArray<u8, N>`
// conversion (the only GA-related path slot 98/99 didn't cover — they
// constructed via `GenericArray::default()` instead).
#[inline(never)]
pub fn check_field_bytes_into_conversion() -> u32 {
    use k256::elliptic_curve::FieldBytes;
    let arr: [u8; 32] = SECP256K1_GX_BYTES;
    let arr = core::hint::black_box(arr);
    let ga: &FieldBytes<k256::Secp256k1> = (&arr).into();
    let first = ga[0];
    let last = ga[31];
    (first == 0x79 && last == 0x98) as u32
}

// Slot 110: `dst_ga.copy_from_slice(src_ga)` where source IS a
// `&GenericArray<u8, U32>` (not `&[u8; 32]`). Slot 99 already covered
// `&[u8; 32]` source. The function `EncodedPoint::from_affine_coordinates`
// uses `bytes[1..33].copy_from_slice(x)` where `x: &GenericArray`, so
// the source-side Deref→slice conversion happens implicitly.
#[inline(never)]
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
