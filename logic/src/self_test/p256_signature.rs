//! p256 signature self-tests: primitives, pipeline stages, and regressions.
use super::fixtures::*;
use super::record_candidate;
use core::hint::black_box;

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

fn self_test_digest_p256_signature() -> [u8; 32] {
    use crate::crypto::sha256::Sha256;
    use crate::{modes::p256_signature_vanity::*, search::hex_pattern::HexPattern};
    let mut h = Sha256::new();
    let mut private = [0; 32];
    private[31] = 1;
    let public = crate::crypto::p256_vanity::public_point(&black_box(private)).unwrap();
    let message = black_box(b"header\0\0footer");
    for source in [0, 1] {
        for target in [0, 1, 2] {
            for s_form in [0, 1, 2] {
                let request = black_box(P256SignatureRequest {
                    private,
                    seed: [0x42; 32],
                    fingerprint: Sha256::digest(public),
                    digest: Sha256::digest(message),
                    worker: 5,
                    offset: 6,
                    length: 2,
                    source,
                    target,
                    s_form,
                    reserved: 0,
                });
                let pattern =
                    black_box(HexPattern::new("", "", if target == 0 { 64 } else { 32 }).unwrap());
                for counter in 254..258 {
                    record_candidate(
                        &mut h,
                        p256_signature(&request, message, black_box(counter), &pattern),
                    );
                }
            }
        }
    }
    h.finalize()
}

pub fn check_p256_signature_end_to_end() -> u32 {
    u32::from(
        self_test_digest_p256_signature()
            == [
                170, 58, 134, 146, 246, 219, 56, 192, 116, 136, 47, 171, 27, 209, 142, 48, 188,
                149, 143, 149, 122, 57, 47, 14, 102, 209, 87, 53, 245, 131, 66, 177,
            ],
    )
}
