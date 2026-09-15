//! rsa pss self-tests: primitives, pipeline stages, and regressions.
use super::fixtures::*;
use super::record_candidate;
use core::hint::black_box;

pub fn check_rsa_pss_sha256() -> u32 {
    u32::from((|| {
        use crate::crypto::sha256::Sha256;
        let actual: [u8; 32] = Sha256::digest(black_box(b"sample"));
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

fn self_test_digest_rsa_pss() -> [u8; 32] {
    use crate::crypto::sha256::Sha256;
    use crate::{modes::rsa_pss_signature_vanity::*, search::hex_pattern::HexPattern};
    let mut h = Sha256::new();
    let message = black_box(b"header\0\0footer");
    for source in [0, 1] {
        for length in [0, 1, 32, 222] {
            let request = black_box(RsaPssRequest {
                p: SELF_TEST_RSA_P,
                q: SELF_TEST_RSA_Q,
                dp: SELF_TEST_RSA_DP,
                dq: SELF_TEST_RSA_DQ,
                q_inv: SELF_TEST_RSA_Q_INV,
                digest: Sha256::digest(message),
                salt: [255; 222],
                reserved: [0; 2],
                offset: 6,
                length: 2,
                source,
                salt_length: length,
            });
            let pattern = black_box(HexPattern::new("", "", 256).unwrap());
            let count = if length == 0 && source == 0 { 1 } else { 4 };
            for counter in 0..count {
                record_candidate(
                    &mut h,
                    rsa_pss(&request, message, black_box(counter), &pattern),
                );
            }
        }
    }
    h.finalize()
}

pub fn check_rsa_pss_end_to_end() -> u32 {
    u32::from(
        self_test_digest_rsa_pss()
            == [
                186, 112, 218, 248, 118, 160, 144, 0, 191, 165, 67, 7, 165, 196, 6, 70, 218, 174,
                58, 193, 60, 84, 214, 84, 233, 131, 204, 111, 141, 86, 47, 85,
            ],
    )
}
