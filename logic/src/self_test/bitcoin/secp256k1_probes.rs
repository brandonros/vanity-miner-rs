//! Concrete secp256k1 probes used by this mode's device self-test.
use super::*;

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

pub fn check_k256_derive_scalar_one() -> u32 {
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 1;
    let pub_key = secp256k1_derive_public_key(&priv_bytes);
    (pub_key == SECP256K1_GENERATOR_COMPRESSED) as u32
}

pub fn check_k256_derive_scalar_two() -> u32 {
    let mut priv_bytes = [0u8; 32];
    priv_bytes[31] = 2;
    let pub_key = secp256k1_derive_public_key(&priv_bytes);
    (pub_key == SECP256K1_TWO_G_COMPRESSED) as u32
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
