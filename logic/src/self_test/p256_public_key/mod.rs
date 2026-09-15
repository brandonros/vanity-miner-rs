//! p256 public key self-tests: primitives, pipeline stages, and regressions.
mod fixtures;
use super::known_answers::*;
use super::record_candidate;
use core::hint::black_box;
use fixtures::*;

#[inline(never)]
pub fn check_p256_public_key_hmac_derivation() -> u32 {
    u32::from((|| {
        let d = crate::search::candidate_derivation::CandidateDeriver::new(
            black_box([0x42; 32]),
            crate::search::candidate_derivation::CandidateDomain::P256PrivateKey,
            [0; 32],
            [0; 32],
        );
        d.block(black_box(3), black_box(4), 0) == CRYPTO_FIXTURE_PRIVATE_DERIVATION
    })())
}

#[inline(never)]
pub fn check_p256_public_key_scalar_derivation() -> u32 {
    u32::from((|| {
        let d = crate::search::candidate_derivation::CandidateDeriver::new(
            black_box([0x42; 32]),
            crate::search::candidate_derivation::CandidateDomain::P256PrivateKey,
            [0; 32],
            [0; 32],
        );
        crate::crypto::p256::candidate_scalar(&d, black_box(3), black_box(4))
            .is_some_and(|s| *s == CRYPTO_FIXTURE_PRIVATE_DERIVATION)
    })())
}

#[inline(never)]
pub fn check_p256_public_key_generator() -> u32 {
    u32::from((|| {
        let mut scalar = [0; 32];
        scalar[31] = 1;
        crate::crypto::p256::public_point(&black_box(scalar)) == Some(CRYPTO_FIXTURE_P256_GENERATOR)
    })())
}

#[inline(never)]
pub fn check_p256_public_key_point_double() -> u32 {
    u32::from((|| {
        let mut scalar = [0; 32];
        scalar[31] = 2;
        crate::crypto::p256::public_point(&black_box(scalar)) == Some(CRYPTO_FIXTURE_P256_DOUBLE)
    })())
}

#[inline(never)]
pub fn check_p256_public_key_zero_scalar_rejected() -> u32 {
    u32::from((|| {
        crate::crypto::p256::public_point(&black_box([0; 32])).is_none()
    })())
}

#[inline(never)]
pub fn check_p256_public_key_order_scalar_rejected() -> u32 {
    u32::from((|| {
        crate::crypto::p256::public_point(&black_box(CRYPTO_FIXTURE_P256_ORDER)).is_none()
    })())
}

#[inline(never)]
pub fn check_p256_public_key_x_encoding() -> u32 {
    u32::from((|| {
        crate::crypto::p256::PublicTarget::X.bytes(&black_box(CRYPTO_FIXTURE_P256_GENERATOR))
            == &CRYPTO_FIXTURE_P256_GENERATOR[1..33]
    })())
}

#[inline(never)]
pub fn check_p256_public_key_y_encoding() -> u32 {
    u32::from((|| {
        crate::crypto::p256::PublicTarget::Y.bytes(&black_box(CRYPTO_FIXTURE_P256_GENERATOR))
            == &CRYPTO_FIXTURE_P256_GENERATOR[33..65]
    })())
}

fn self_test_digest_p256_public() -> [u8; 32] {
    use crate::crypto::sha256::Sha256;
    use crate::{modes::p256_public_key::*, search::hex_pattern::HexPattern};
    let mut h = Sha256::new();
    for (target, width) in [(0, 32), (1, 32), (2, 64), (3, 65)] {
        let request = black_box(P256PublicRequest {
            seed: [0x42; 32],
            worker: 7,
            target,
            reserved: 0,
        });
        for prefix in ["", "f"] {
            let pattern = black_box(HexPattern::new(prefix, "", width).unwrap());
            for counter in u64::MAX - 7..=u64::MAX {
                record_candidate(&mut h, p256_public(&request, black_box(counter), &pattern));
            }
        }
    }
    h.finalize()
}

#[inline(never)]
pub fn check_p256_public_end_to_end() -> u32 {
    u32::from(
        self_test_digest_p256_public()
            == [
                239, 76, 14, 206, 113, 184, 37, 233, 82, 252, 160, 108, 166, 113, 26, 230, 132, 89,
                110, 201, 235, 102, 51, 125, 141, 160, 109, 133, 191, 253, 11, 243,
            ],
    )
}

/// Write only this mode's stable result slots.
pub fn run(results: &mut [u32]) {
    results[118] = check_p256_public_key_hmac_derivation();
    results[119] = check_p256_public_key_scalar_derivation();
    results[120] = check_p256_public_key_generator();
    results[121] = check_p256_public_key_point_double();
    results[122] = check_p256_public_key_zero_scalar_rejected();
    results[123] = check_p256_public_key_order_scalar_rejected();
    results[124] = check_p256_public_key_x_encoding();
    results[125] = check_p256_public_key_y_encoding();
    results[153] = check_p256_public_end_to_end();
}
