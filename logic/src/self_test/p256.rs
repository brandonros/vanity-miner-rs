//! P-256 primitive, representation and boundary checks.
use super::fixtures::*;
use core::hint::black_box;
pub fn check_p256_public_key_hmac_derivation() -> u32 {
    u32::from((|| {
        let d = crate::search::crypto_search::CandidateDeriver::new(
            black_box([0x42; 32]),
            crate::search::crypto_search::CandidateDomain::P256PrivateKey,
            [0; 32],
            [0; 32],
        );
        d.block(black_box(3), black_box(4), 0) == CRYPTO_FIXTURE_PRIVATE_DERIVATION
    })())
}

pub fn check_p256_public_key_scalar_derivation() -> u32 {
    u32::from((|| {
        let d = crate::search::crypto_search::CandidateDeriver::new(
            black_box([0x42; 32]),
            crate::search::crypto_search::CandidateDomain::P256PrivateKey,
            [0; 32],
            [0; 32],
        );
        crate::crypto::p256_vanity::candidate_scalar(&d, black_box(3), black_box(4))
            .is_some_and(|s| *s == CRYPTO_FIXTURE_PRIVATE_DERIVATION)
    })())
}

pub fn check_p256_public_key_generator() -> u32 {
    u32::from((|| {
        let mut scalar = [0; 32];
        scalar[31] = 1;
        crate::crypto::p256_vanity::public_point(&black_box(scalar))
            == Some(CRYPTO_FIXTURE_P256_GENERATOR)
    })())
}

pub fn check_p256_public_key_point_double() -> u32 {
    u32::from((|| {
        let mut scalar = [0; 32];
        scalar[31] = 2;
        crate::crypto::p256_vanity::public_point(&black_box(scalar))
            == Some(CRYPTO_FIXTURE_P256_DOUBLE)
    })())
}

pub fn check_p256_public_key_zero_scalar_rejected() -> u32 {
    u32::from((|| {
        crate::crypto::p256_vanity::public_point(&black_box([0; 32])).is_none()
    })())
}

pub fn check_p256_public_key_order_scalar_rejected() -> u32 {
    u32::from((|| {
        crate::crypto::p256_vanity::public_point(&black_box(CRYPTO_FIXTURE_P256_ORDER)).is_none()
    })())
}

pub fn check_p256_public_key_x_encoding() -> u32 {
    u32::from((|| {
        crate::crypto::p256_vanity::PublicTarget::X.bytes(&black_box(CRYPTO_FIXTURE_P256_GENERATOR))
            == &CRYPTO_FIXTURE_P256_GENERATOR[1..33]
    })())
}

pub fn check_p256_public_key_y_encoding() -> u32 {
    u32::from((|| {
        crate::crypto::p256_vanity::PublicTarget::Y.bytes(&black_box(CRYPTO_FIXTURE_P256_GENERATOR))
            == &CRYPTO_FIXTURE_P256_GENERATOR[33..65]
    })())
}

pub fn check_p256_signature_rfc6979_sample() -> u32 {
    u32::from((|| {
        crate::crypto::p256_vanity::signatures::sign_message(
            &black_box(CRYPTO_FIXTURE_RFC6979_KEY),
            black_box(b"sample"),
        ) == Some(CRYPTO_FIXTURE_RFC6979_SAMPLE)
    })())
}

pub fn check_p256_signature_rfc6979_test() -> u32 {
    u32::from((|| {
        crate::crypto::p256_vanity::signatures::sign_message(
            &black_box(CRYPTO_FIXTURE_RFC6979_KEY),
            black_box(b"test"),
        ) == Some(CRYPTO_FIXTURE_RFC6979_TEST)
    })())
}

pub fn check_p256_signature_ephemeral_r() -> u32 {
    u32::from((|| {
        let mut nonce = [0; 32];
        nonce[31] = 1;
        crate::crypto::p256_vanity::signatures::ephemeral_r(&black_box(nonce))
            .is_some_and(|r| r.as_slice() == &CRYPTO_FIXTURE_P256_GENERATOR[1..33])
    })())
}

pub fn check_p256_signature_ephemeral_signature() -> u32 {
    u32::from((|| {
        let mut one = [0; 32];
        one[31] = 1;
        crate::crypto::p256_vanity::signatures::sign_digest_ephemeral(
            &black_box(one),
            &black_box(CRYPTO_FIXTURE_SAMPLE_SHA256),
            &black_box(one),
        ) == Some(CRYPTO_FIXTURE_EPHEMERAL_ONE)
    })())
}

pub fn check_p256_signature_zero_nonce_rejected() -> u32 {
    u32::from((|| {
        crate::crypto::p256_vanity::signatures::sign_digest_ephemeral(
            &black_box(CRYPTO_FIXTURE_RFC6979_KEY),
            &black_box(CRYPTO_FIXTURE_SAMPLE_SHA256),
            &black_box([0; 32]),
        )
        .is_none()
    })())
}

pub fn check_p256_signature_low_s() -> u32 {
    u32::from((|| {
        use crate::crypto::p256_vanity::signatures::{
            SForm, SignatureTarget, matching_representation,
        };
        let pattern = crate::search::hex_pattern::HexPattern::new("", "", 64).unwrap();
        matching_representation(
            &black_box(CRYPTO_FIXTURE_RFC6979_SAMPLE),
            SignatureTarget::Raw,
            SForm::Low,
            &pattern,
        ) == Some(CRYPTO_FIXTURE_RFC6979_SAMPLE_LOW)
    })())
}

pub fn check_p256_signature_high_s() -> u32 {
    u32::from((|| {
        use crate::crypto::p256_vanity::signatures::{
            SForm, SignatureTarget, matching_representation,
        };
        let pattern = crate::search::hex_pattern::HexPattern::new("", "", 64).unwrap();
        matching_representation(
            &black_box(CRYPTO_FIXTURE_RFC6979_SAMPLE),
            SignatureTarget::Raw,
            SForm::High,
            &pattern,
        ) == Some(CRYPTO_FIXTURE_RFC6979_SAMPLE)
    })())
}

pub fn check_p256_signature_message_window_carry() -> u32 {
    u32::from((|| {
        crate::search::crypto_search::hash_message_counter(
            black_box(b"header\0\0footer"),
            6,
            2,
            black_box(256),
        ) == Ok(CRYPTO_FIXTURE_WINDOW_SHA256)
    })())
}

pub fn check_p256_signature_ephemeral_hmac() -> u32 {
    u32::from((|| {
        let d = crate::search::crypto_search::CandidateDeriver::new(
            black_box([0x42; 32]),
            crate::search::crypto_search::CandidateDomain::P256Ephemeral,
            [1; 32],
            [2; 32],
        );
        d.block(black_box(3), black_box(4), black_box(5)) == CRYPTO_FIXTURE_EPHEMERAL_DERIVATION
    })())
}

#[cfg(test)]
mod p256_public_key_tests {
    #[test]
    fn hmac_derivation() {
        assert_eq!(super::check_p256_public_key_hmac_derivation(), 1);
    }
    #[test]
    fn scalar_derivation() {
        assert_eq!(super::check_p256_public_key_scalar_derivation(), 1);
    }
    #[test]
    fn generator() {
        assert_eq!(super::check_p256_public_key_generator(), 1);
    }
    #[test]
    fn point_double() {
        assert_eq!(super::check_p256_public_key_point_double(), 1);
    }
    #[test]
    fn zero_scalar_rejected() {
        assert_eq!(super::check_p256_public_key_zero_scalar_rejected(), 1);
    }
    #[test]
    fn order_scalar_rejected() {
        assert_eq!(super::check_p256_public_key_order_scalar_rejected(), 1);
    }
    #[test]
    fn x_encoding() {
        assert_eq!(super::check_p256_public_key_x_encoding(), 1);
    }
    #[test]
    fn y_encoding() {
        assert_eq!(super::check_p256_public_key_y_encoding(), 1);
    }
}

#[cfg(test)]
mod p256_signature_tests {
    #[test]
    fn rfc6979_sample() {
        assert_eq!(super::check_p256_signature_rfc6979_sample(), 1);
    }
    #[test]
    fn rfc6979_test() {
        assert_eq!(super::check_p256_signature_rfc6979_test(), 1);
    }
    #[test]
    fn ephemeral_r() {
        assert_eq!(super::check_p256_signature_ephemeral_r(), 1);
    }
    #[test]
    fn ephemeral_signature() {
        assert_eq!(super::check_p256_signature_ephemeral_signature(), 1);
    }
    #[test]
    fn zero_nonce_rejected() {
        assert_eq!(super::check_p256_signature_zero_nonce_rejected(), 1);
    }
    #[test]
    fn low_s() {
        assert_eq!(super::check_p256_signature_low_s(), 1);
    }
    #[test]
    fn high_s() {
        assert_eq!(super::check_p256_signature_high_s(), 1);
    }
    #[test]
    fn message_window_carry() {
        assert_eq!(super::check_p256_signature_message_window_carry(), 1);
    }
    #[test]
    fn ephemeral_hmac() {
        assert_eq!(super::check_p256_signature_ephemeral_hmac(), 1);
    }
}
