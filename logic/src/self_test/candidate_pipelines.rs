//! Full candidate pipelines compared with fixed CPU reference digests.
use super::fixtures::*;
use core::hint::black_box;
fn record_candidate(
    h: &mut sha2::Sha256,
    result: crate::search::candidate_result::CandidateResult,
) {
    use sha2::Digest;
    h.update(result.status.to_le_bytes());
    h.update(result.bytes);
}

fn self_test_digest_p256_public() -> [u8; 32] {
    use crate::{modes::p256_public_key_vanity::*, search::hex_pattern::HexPattern};
    use sha2::{Digest, Sha256};
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
    h.finalize().into()
}

fn self_test_digest_p256_signature() -> [u8; 32] {
    use crate::{modes::p256_signature_vanity::*, search::hex_pattern::HexPattern};
    use sha2::{Digest, Sha256};
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
                    fingerprint: Sha256::digest(public).into(),
                    digest: Sha256::digest(message).into(),
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
    h.finalize().into()
}

fn self_test_digest_rsa_pss() -> [u8; 32] {
    use crate::{modes::rsa_pss_signature_vanity::*, search::hex_pattern::HexPattern};
    use sha2::{Digest, Sha256};
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
                digest: Sha256::digest(message).into(),
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
    h.finalize().into()
}

fn self_test_digest_rsa_modulus() -> [u8; 32] {
    use crate::{modes::rsa_modulus_vanity::*, search::hex_pattern::HexPattern};
    use crypto_bigint::{Encoding, U1024};
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let q = U1024::from_be_slice(&black_box(SELF_TEST_RSA_Q));
    let request = black_box(RsaModulusRequest {
        p: SELF_TEST_RSA_P,
        first: SELF_TEST_RSA_Q,
        stride: U1024::from_u64(2).to_be_bytes(),
        upper: q.wrapping_add(&U1024::from_u64(8)).to_be_bytes(),
    });
    let pattern = black_box(HexPattern::new("", "", 256).unwrap());
    for counter in 0..6 {
        record_candidate(&mut h, rsa_modulus(&request, black_box(counter), &pattern));
    }
    h.finalize().into()
}

pub fn check_p256_public_end_to_end() -> u32 {
    u32::from(
        self_test_digest_p256_public()
            == [
                239, 76, 14, 206, 113, 184, 37, 233, 82, 252, 160, 108, 166, 113, 26, 230, 132, 89,
                110, 201, 235, 102, 51, 125, 141, 160, 109, 133, 191, 253, 11, 243,
            ],
    )
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

pub fn check_rsa_pss_end_to_end() -> u32 {
    u32::from(
        self_test_digest_rsa_pss()
            == [
                186, 112, 218, 248, 118, 160, 144, 0, 191, 165, 67, 7, 165, 196, 6, 70, 218, 174,
                58, 193, 60, 84, 214, 84, 233, 131, 204, 111, 141, 86, 47, 85,
            ],
    )
}

pub fn check_rsa_modulus_end_to_end() -> u32 {
    u32::from(
        self_test_digest_rsa_modulus()
            == [
                25, 124, 199, 181, 191, 98, 176, 178, 157, 56, 112, 118, 55, 67, 54, 139, 52, 12,
                129, 41, 1, 171, 41, 0, 200, 15, 176, 149, 51, 55, 253, 50,
            ],
    )
}
