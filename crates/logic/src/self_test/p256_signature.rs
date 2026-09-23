//! p256 signature: RFC 6979, ephemeral signatures, s normalization, message
//! windows, nonce derivation, and the candidate pipeline.
use super::{P256_GENERATOR, SAMPLE_SHA256, record_candidate};
use crate::crypto::p256::public_point;
use crate::crypto::p256::signatures::{
    self, SForm, SignatureTarget, matching_representation, sign_digest_ephemeral, sign_message,
};
use crate::crypto::sha256::Sha256;
use crate::modes::p256_signature::{P256SignatureRequest, p256_signature};
use crate::search::candidate_derivation::{CandidateDeriver, CandidateDomain};
use crate::search::hex_pattern::HexPattern;
use crate::search::message_window::{WindowError, hash_message_counter, write_message_counter};
use core::hint::black_box;

const EPHEMERAL_DERIVATION: [u8; 32] = [
    0x77, 0x9c, 0x97, 0x6e, 0x69, 0xd7, 0x16, 0x4a, 0xf4, 0x81, 0x53, 0x0f, 0x0f, 0xc2, 0x2b, 0x27,
    0x8c, 0xdc, 0x23, 0x8e, 0x08, 0x35, 0x35, 0xa6, 0xdd, 0x18, 0x99, 0x42, 0xdc, 0x84, 0xe0, 0xaa,
];
const RFC6979_KEY: [u8; 32] = [
    0xc9, 0xaf, 0xa9, 0xd8, 0x45, 0xba, 0x75, 0x16, 0x6b, 0x5c, 0x21, 0x57, 0x67, 0xb1, 0xd6, 0x93,
    0x4e, 0x50, 0xc3, 0xdb, 0x36, 0xe8, 0x9b, 0x12, 0x7b, 0x8a, 0x62, 0x2b, 0x12, 0x0f, 0x67, 0x21,
];
const RFC6979_SAMPLE: [u8; 64] = [
    0xef, 0xd4, 0x8b, 0x2a, 0xac, 0xb6, 0xa8, 0xfd, 0x11, 0x40, 0xdd, 0x9c, 0xd4, 0x5e, 0x81, 0xd6,
    0x9d, 0x2c, 0x87, 0x7b, 0x56, 0xaa, 0xf9, 0x91, 0xc3, 0x4d, 0x0e, 0xa8, 0x4e, 0xaf, 0x37, 0x16,
    0xf7, 0xcb, 0x1c, 0x94, 0x2d, 0x65, 0x7c, 0x41, 0xd4, 0x36, 0xc7, 0xa1, 0xb6, 0xe2, 0x9f, 0x65,
    0xf3, 0xe9, 0x00, 0xdb, 0xb9, 0xaf, 0xf4, 0x06, 0x4d, 0xc4, 0xab, 0x2f, 0x84, 0x3a, 0xcd, 0xa8,
];
const RFC6979_TEST: [u8; 64] = [
    0xf1, 0xab, 0xb0, 0x23, 0x51, 0x83, 0x51, 0xcd, 0x71, 0xd8, 0x81, 0x56, 0x7b, 0x1e, 0xa6, 0x63,
    0xed, 0x3e, 0xfc, 0xf6, 0xc5, 0x13, 0x2b, 0x35, 0x4f, 0x28, 0xd3, 0xb0, 0xb7, 0xd3, 0x83, 0x67,
    0x01, 0x9f, 0x41, 0x13, 0x74, 0x2a, 0x2b, 0x14, 0xbd, 0x25, 0x92, 0x6b, 0x49, 0xc6, 0x49, 0x15,
    0x5f, 0x26, 0x7e, 0x60, 0xd3, 0x81, 0x4b, 0x4c, 0x0c, 0xc8, 0x42, 0x50, 0xe4, 0x6f, 0x00, 0x83,
];
const RFC6979_SAMPLE_LOW: [u8; 64] = [
    0xef, 0xd4, 0x8b, 0x2a, 0xac, 0xb6, 0xa8, 0xfd, 0x11, 0x40, 0xdd, 0x9c, 0xd4, 0x5e, 0x81, 0xd6,
    0x9d, 0x2c, 0x87, 0x7b, 0x56, 0xaa, 0xf9, 0x91, 0xc3, 0x4d, 0x0e, 0xa8, 0x4e, 0xaf, 0x37, 0x16,
    0x08, 0x34, 0xe3, 0x6a, 0xd2, 0x9a, 0x83, 0xbf, 0x2b, 0xc9, 0x38, 0x5e, 0x49, 0x1d, 0x60, 0x99,
    0xc8, 0xfd, 0xf9, 0xd1, 0xed, 0x67, 0xaa, 0x7e, 0xa5, 0xf5, 0x1f, 0x93, 0x78, 0x28, 0x57, 0xa9,
];
const EPHEMERAL_ONE: [u8; 64] = [
    0x6b, 0x17, 0xd1, 0xf2, 0xe1, 0x2c, 0x42, 0x47, 0xf8, 0xbc, 0xe6, 0xe5, 0x63, 0xa4, 0x40, 0xf2,
    0x77, 0x03, 0x7d, 0x81, 0x2d, 0xeb, 0x33, 0xa0, 0xf4, 0xa1, 0x39, 0x45, 0xd8, 0x98, 0xc2, 0x96,
    0x1a, 0x43, 0xad, 0xd5, 0x8b, 0xc7, 0xb1, 0x08, 0xdb, 0x6a, 0xc8, 0xbb, 0xf8, 0x98, 0x60, 0xb9,
    0xd4, 0x9f, 0x9f, 0xd5, 0xef, 0xbd, 0x1e, 0x31, 0x62, 0xf8, 0xac, 0x0d, 0x3e, 0xe3, 0x6f, 0x04,
];
const WINDOW_SHA256: [u8; 32] = [
    0xf0, 0x40, 0x94, 0xb1, 0xc0, 0x3a, 0x68, 0x4e, 0x07, 0xaa, 0x9e, 0xd3, 0xfc, 0xb9, 0x05, 0x96,
    0xaf, 0xf3, 0x93, 0x8e, 0x31, 0xcd, 0xfd, 0x84, 0x2c, 0x7b, 0xbb, 0xdd, 0x2c, 0xe2, 0xca, 0xd2,
];
const LONG_WINDOW_SHA256: [u8; 32] = [
    0xeb, 0xbb, 0xde, 0xf7, 0x72, 0x33, 0xb5, 0xb0, 0xb8, 0x87, 0xff, 0x91, 0x8e, 0x76, 0x4a, 0x4a,
    0x5d, 0x30, 0x72, 0x15, 0x02, 0xe2, 0xcf, 0x2a, 0x0d, 0xed, 0xca, 0xe5, 0x11, 0x2c, 0xb1, 0xb9,
];

/// The RFC 6979 sample signature in one s form.
fn representation(s_form: SForm) -> Option<[u8; 64]> {
    let pattern = HexPattern::new("", "", 64).unwrap();
    matching_representation(
        &black_box(RFC6979_SAMPLE),
        SignatureTarget::Raw,
        s_form,
        &pattern,
    )
}

fn digest() -> [u8; 32] {
    let mut h = Sha256::new();
    let mut private = [0; 32];
    private[31] = 1;
    let public = public_point(&black_box(private)).unwrap();
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

checks! {
    /// p256 signature rfc6979 sample
    fn rfc6979_sample() -> bool {
        sign_message(&black_box(RFC6979_KEY), black_box(b"sample")) == Some(RFC6979_SAMPLE)
    }

    /// p256 signature rfc6979 test
    fn rfc6979_test() -> bool {
        sign_message(&black_box(RFC6979_KEY), black_box(b"test")) == Some(RFC6979_TEST)
    }

    /// p256 signature ephemeral r
    fn ephemeral_r() -> bool {
        let mut nonce = [0; 32];
        nonce[31] = 1;
        signatures::ephemeral_r(&black_box(nonce)).is_some_and(|r| r.as_slice() == &P256_GENERATOR[1..33])
    }

    /// p256 signature ephemeral signature
    fn ephemeral_signature() -> bool {
        let mut one = [0; 32];
        one[31] = 1;
        sign_digest_ephemeral(&black_box(one), &black_box(SAMPLE_SHA256), &black_box(one))
            == Some(EPHEMERAL_ONE)
    }

    /// p256 signature zero nonce rejected
    fn zero_nonce_rejected() -> bool {
        sign_digest_ephemeral(
            &black_box(RFC6979_KEY),
            &black_box(SAMPLE_SHA256),
            &black_box([0; 32]),
        )
        .is_none()
    }

    /// p256 signature low s
    fn low_s() -> bool {
        representation(SForm::Low) == Some(RFC6979_SAMPLE_LOW)
    }

    /// p256 signature high s
    fn high_s() -> bool {
        representation(SForm::High) == Some(RFC6979_SAMPLE)
    }

    /// p256 signature message window carry
    fn message_window_carry() -> bool {
        hash_message_counter(black_box(b"header\0\0footer"), 6, 2, black_box(256))
            == Ok(WINDOW_SHA256)
    }

    /// p256 signature ephemeral hmac
    fn ephemeral_hmac() -> bool {
        let d = CandidateDeriver::new(
            black_box([0x42; 32]),
            CandidateDomain::P256Ephemeral,
            [1; 32],
            [2; 32],
        );
        d.block(black_box(3), black_box(4), black_box(5)) == EPHEMERAL_DERIVATION
    }

    /// end-to-end p256 signature candidate pipeline
    fn end_to_end() -> bool {
        digest()
            == [
                170, 58, 134, 146, 246, 219, 56, 192, 116, 136, 47, 171, 27, 209, 142, 48, 188,
                149, 143, 149, 122, 57, 47, 14, 102, 209, 87, 53, 245, 131, 66, 177,
            ]
    }

    /// message counter last value and exhaustion
    fn message_counter_exhaustion() -> bool {
        let mut message = black_box([0xa5; 4]);
        if write_message_counter(&mut message, black_box(1), black_box(1), black_box(255)) != Ok(())
            || message != [0xa5, 255, 0xa5, 0xa5]
        {
            return false;
        }
        write_message_counter(&mut message, black_box(1), black_box(1), black_box(256))
            == Err(WindowError::Exhausted)
            && message == [0xa5, 255, 0xa5, 0xa5]
            && hash_message_counter(&message, black_box(1), black_box(1), black_box(256))
                == Err(WindowError::Exhausted)
    }

    /// message window rejects invalid bounds without mutation
    fn message_window_invalid() -> bool {
        for (offset, length) in [(0, 0), (4, 1), (usize::MAX, 2)] {
            let mut message = black_box([0xa5; 4]);
            let offset = black_box(offset);
            let length = black_box(length);
            if write_message_counter(&mut message, offset, length, black_box(0))
                != Err(WindowError::InvalidBounds)
                || message != [0xa5; 4]
                || hash_message_counter(&message, offset, length, black_box(0))
                    != Err(WindowError::InvalidBounds)
            {
                return false;
            }
        }
        true
    }

    /// message window across multiple sha256 blocks
    fn message_window_multiblock() -> bool {
        let message = black_box([0xa5; 160]);
        let counter = black_box(0x0102030405060708090a0b0c0d0e0f10u128);
        let mut materialized = message;
        if write_message_counter(&mut materialized, black_box(7), black_box(129), counter) != Ok(()) {
            return false;
        }
        // Independently specified bytes, including unchanged prefix/suffix and all zero padding.
        let mut expected = [0xa5; 160];
        expected[7..120].fill(0);
        expected[120..136].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        materialized == expected
            && hash_message_counter(&message, black_box(7), black_box(129), counter)
                == Ok(LONG_WINDOW_SHA256)
    }
}
