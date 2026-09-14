//! RSA arithmetic, PSS and boundary checks.
use super::fixtures::*;
use core::hint::black_box;
pub fn check_rsa_pss_sha256() -> u32 {
    u32::from((|| {
        use sha2::{Digest, Sha256};
        let actual: [u8; 32] = Sha256::digest(black_box(b"sample")).into();
        actual == CRYPTO_FIXTURE_SAMPLE_SHA256
    })())
}

pub fn check_rsa_pss_mgf1_partial_block() -> u32 {
    u32::from((|| {
        let mut out = [0; 50];
        crate::crypto::rsa_pss::mgf1_sha256(black_box(b"public test seed"), &mut out).is_ok()
            && out == CRYPTO_FIXTURE_MGF_PARTIAL
    })())
}

pub fn check_rsa_pss_salt32_encoding() -> u32 {
    u32::from((|| {
        let salt = black_box(core::array::from_fn::<_, 32, _>(|i| i as u8));
        let mut out = [0; 256];
        crate::crypto::rsa_pss::encode_sha256(
            &black_box(CRYPTO_FIXTURE_SAMPLE_SHA256),
            &salt,
            2047,
            &mut out,
        )
        .is_ok()
            && out == CRYPTO_FIXTURE_PSS_SALT32
    })())
}

pub fn check_rsa_pss_empty_salt_encoding() -> u32 {
    u32::from((|| {
        let salt = black_box([0u8; 0]);
        let mut out = [0; 256];
        crate::crypto::rsa_pss::encode_sha256(
            &black_box(CRYPTO_FIXTURE_SAMPLE_SHA256),
            &salt,
            2047,
            &mut out,
        )
        .is_ok()
            && out == CRYPTO_FIXTURE_PSS_EMPTY_SALT
    })())
}

pub fn check_rsa_pss_maximum_salt_encoding() -> u32 {
    u32::from((|| {
        let salt = black_box([0x42; 222]);
        let mut out = [0; 256];
        crate::crypto::rsa_pss::encode_sha256(
            &black_box(CRYPTO_FIXTURE_SAMPLE_SHA256),
            &salt,
            2047,
            &mut out,
        )
        .is_ok()
            && out == CRYPTO_FIXTURE_PSS_MAX_SALT
    })())
}

pub fn check_rsa_pss_oversized_salt_rejected() -> u32 {
    u32::from((|| {
        let mut out = [0xa5; 256];
        crate::crypto::rsa_pss::encode_sha256(
            &black_box(CRYPTO_FIXTURE_SAMPLE_SHA256),
            &black_box([0; 223]),
            2047,
            &mut out,
        ) == Err(crate::crypto::rsa_pss::PssError::SaltTooLong)
            && out == [0xa5; 256]
    })())
}

pub fn check_rsa_pss_salt_carry() -> u32 {
    u32::from((|| {
        let mut out = [0; 2];
        crate::search::crypto_search::write_salt_counter(
            &black_box([0xff; 2]),
            black_box(1),
            &mut out,
        )
        .is_ok()
            && out == [0; 2]
    })())
}

pub fn check_rsa_pss_crt_known_answer() -> u32 {
    let key = self_test_crt_key();
    let mut input = [0; 256];
    input[255] = 65;
    u32::from(key.private_operation(&black_box(input)) == Some(SELF_TEST_RSA_SIGNATURE_65))
}

pub fn check_rsa_pss_crt_fault_rejected() -> u32 {
    // Deliberately composite p simulates inconsistent CRT arithmetic while
    // leaving constructor congruence checks satisfied. The final public-operation
    // verification must reject the result. No private fields or test hooks needed.
    let key = crate::crypto::rsa_crt::Rsa2048Crt::new(
        &black_box([255; 128]),
        &black_box(SELF_TEST_RSA_Q),
        &black_box(SELF_TEST_COMPOSITE_DP),
        &black_box(SELF_TEST_RSA_DQ),
        &black_box(SELF_TEST_COMPOSITE_Q_INV),
    )
    .unwrap();
    let mut input = [0; 256];
    input[255] = 65;
    u32::from(key.private_operation(&black_box(input)).is_none())
}

pub fn check_rsa_pss_crt_modulus_rejected() -> u32 {
    let key = self_test_crt_key();
    u32::from(key.private_operation(&black_box(key.modulus())).is_none())
}

pub fn check_rsa_modulus_multiplication_carry() -> u32 {
    u32::from((|| {
        use crypto_bigint::{Encoding, U1024, U2048};
        let x = black_box(U1024::MAX);
        let result: U2048 = x.mul(&x);
        result.to_be_bytes() == CRYPTO_FIXTURE_MUL_CARRY
    })())
}

pub fn check_rsa_modulus_progression_carry() -> u32 {
    u32::from((|| {
        use crypto_bigint::{Encoding, U1024, U2048};
        let first = black_box(U1024::MAX);
        let stride = black_box(U1024::from_u8(2));
        let step: U2048 = stride.mul(&black_box(U1024::ONE));
        step.wrapping_add(&first.resize()).to_be_bytes() == CRYPTO_FIXTURE_PROGRESSION_CARRY
    })())
}

pub fn check_rsa_modulus_prime_filter() -> u32 {
    u32::from((|| {
        crate::crypto::rsa_prime::probable_prime(&black_box(crypto_bigint::U1024::from_u64(104729)))
    })())
}

pub fn check_rsa_modulus_pseudoprime_rejected() -> u32 {
    u32::from((|| {
        !crate::crypto::rsa_prime::probable_prime(&black_box(crypto_bigint::U1024::from_u64(
            341550071728321,
        )))
    })())
}

pub fn check_rsa_modulus_zero_stride_rejected() -> u32 {
    u32::from((|| {
        use crate::modes::rsa_modulus_vanity::{RsaModulusRequest, rsa_modulus};
        let mut request = RsaModulusRequest {
            p: [0; 128],
            first: [0; 128],
            stride: [0; 128],
            upper: [0xff; 128],
        };
        request.p[0] = 0xc0;
        request.first[0] = 0x90;
        request.stride[127] = 2;
        request.stride = [0; 128];
        let pattern = crate::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
        rsa_modulus(&black_box(request), black_box(0), &pattern).status == 2
    })())
}

pub fn check_rsa_modulus_upper_bound_rejected() -> u32 {
    u32::from((|| {
        use crate::modes::rsa_modulus_vanity::{RsaModulusRequest, rsa_modulus};
        let mut request = RsaModulusRequest {
            p: [0; 128],
            first: [0; 128],
            stride: [0; 128],
            upper: [0xff; 128],
        };
        request.p[0] = 0xc0;
        request.first[0] = 0x90;
        request.stride[127] = 2;
        request.upper = [0; 128];
        let pattern = crate::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
        rsa_modulus(&black_box(request), black_box(0), &pattern).status == 2
    })())
}

pub fn check_rsa_modulus_equal_factors_rejected() -> u32 {
    u32::from((|| {
        use crate::modes::rsa_modulus_vanity::{RsaModulusRequest, rsa_modulus};
        let mut request = RsaModulusRequest {
            p: [0; 128],
            first: [0; 128],
            stride: [0; 128],
            upper: [0xff; 128],
        };
        request.p[0] = 0xc0;
        request.first[0] = 0x90;
        request.stride[127] = 2;
        request.first = request.p;
        let pattern = crate::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
        rsa_modulus(&black_box(request), black_box(0), &pattern).status == 0
    })())
}

pub fn check_rsa_modulus_undersized_factor_rejected() -> u32 {
    u32::from((|| {
        use crate::modes::rsa_modulus_vanity::{RsaModulusRequest, rsa_modulus};
        let mut request = RsaModulusRequest {
            p: [0; 128],
            first: [0; 128],
            stride: [0; 128],
            upper: [0xff; 128],
        };
        request.p[0] = 0xc0;
        request.first[0] = 0x90;
        request.stride[127] = 2;
        request.p = [0; 128];
        let pattern = crate::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
        rsa_modulus(&black_box(request), black_box(0), &pattern).status == 0
    })())
}

#[cfg(test)]
mod rsa_pss_tests {
    #[test]
    fn sha256() {
        assert_eq!(super::check_rsa_pss_sha256(), 1);
    }
    #[test]
    fn mgf1_partial_block() {
        assert_eq!(super::check_rsa_pss_mgf1_partial_block(), 1);
    }
    #[test]
    fn salt32_encoding() {
        assert_eq!(super::check_rsa_pss_salt32_encoding(), 1);
    }
    #[test]
    fn empty_salt_encoding() {
        assert_eq!(super::check_rsa_pss_empty_salt_encoding(), 1);
    }
    #[test]
    fn maximum_salt_encoding() {
        assert_eq!(super::check_rsa_pss_maximum_salt_encoding(), 1);
    }
    #[test]
    fn oversized_salt_rejected() {
        assert_eq!(super::check_rsa_pss_oversized_salt_rejected(), 1);
    }
    #[test]
    fn salt_carry() {
        assert_eq!(super::check_rsa_pss_salt_carry(), 1);
    }
    #[test]
    fn crt_known_answer() {
        assert_eq!(super::check_rsa_pss_crt_known_answer(), 1);
    }
    #[test]
    fn crt_fault_rejected() {
        assert_eq!(super::check_rsa_pss_crt_fault_rejected(), 1);
    }
    #[test]
    fn crt_modulus_rejected() {
        assert_eq!(super::check_rsa_pss_crt_modulus_rejected(), 1);
    }
}

#[cfg(test)]
mod rsa_modulus_tests {
    #[test]
    fn multiplication_carry() {
        assert_eq!(super::check_rsa_modulus_multiplication_carry(), 1);
    }
    #[test]
    fn progression_carry() {
        assert_eq!(super::check_rsa_modulus_progression_carry(), 1);
    }
    #[test]
    fn prime_filter() {
        assert_eq!(super::check_rsa_modulus_prime_filter(), 1);
    }
    #[test]
    fn pseudoprime_rejected() {
        assert_eq!(super::check_rsa_modulus_pseudoprime_rejected(), 1);
    }
    #[test]
    fn zero_stride_rejected() {
        assert_eq!(super::check_rsa_modulus_zero_stride_rejected(), 1);
    }
    #[test]
    fn upper_bound_rejected() {
        assert_eq!(super::check_rsa_modulus_upper_bound_rejected(), 1);
    }
    #[test]
    fn equal_factors_rejected() {
        assert_eq!(super::check_rsa_modulus_equal_factors_rejected(), 1);
    }
    #[test]
    fn undersized_factor_rejected() {
        assert_eq!(super::check_rsa_modulus_undersized_factor_rejected(), 1);
    }
}

fn self_test_crt_key() -> crate::crypto::rsa_crt::Rsa2048Crt {
    crate::crypto::rsa_crt::Rsa2048Crt::new(
        &black_box(SELF_TEST_RSA_P),
        &black_box(SELF_TEST_RSA_Q),
        &black_box(SELF_TEST_RSA_DP),
        &black_box(SELF_TEST_RSA_DQ),
        &black_box(SELF_TEST_RSA_Q_INV),
    )
    .unwrap()
}
