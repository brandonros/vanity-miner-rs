//! rsa modulus self-tests: primitives, pipeline stages, and regressions.
use super::fixtures::*;
use super::record_candidate;
use core::hint::black_box;

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
        use crate::modes::rsa_modulus_vanity::{RsaModulusRequest, rsa_modulus};
        let mut stride = [0u8; 128];
        stride[127] = 2;
        let mut request = RsaModulusRequest {
            stage: 1,
            reserved: 0,
            p: [0; 128],
            first: SELF_TEST_RSA_P,
            stride,
            upper: SELF_TEST_RSA_P,
        };
        let pattern = crate::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
        let result = rsa_modulus(&black_box(request), black_box(0), &pattern);
        if result.status != 1 || result.bytes[..128] != SELF_TEST_RSA_P {
            return false;
        }
        if rsa_modulus(&black_box(request), black_box(1), &pattern).status != 2 {
            return false;
        }
        // The prime-filter stage must reject an even 1024-bit candidate.
        request.first[127] &= 0xfe;
        request.upper = request.first;
        rsa_modulus(&black_box(request), black_box(0), &pattern).status == 0
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
            stage: 0,
            reserved: 0,
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
            stage: 0,
            reserved: 0,
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
            stage: 0,
            reserved: 0,
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
            stage: 0,
            reserved: 0,
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

fn self_test_digest_rsa_modulus() -> [u8; 32] {
    use crate::crypto::sha256::Sha256;
    use crate::{modes::rsa_modulus_vanity::*, search::hex_pattern::HexPattern};
    use crypto_bigint::{Encoding, U1024};
    let mut h = Sha256::new();
    let q = U1024::from_be_slice(&black_box(SELF_TEST_RSA_Q));
    let request = black_box(RsaModulusRequest {
        stage: 0,
        reserved: 0,
        p: SELF_TEST_RSA_P,
        first: SELF_TEST_RSA_Q,
        stride: U1024::from_u64(2).to_be_bytes(),
        upper: q.wrapping_add(&U1024::from_u64(8)).to_be_bytes(),
    });
    let pattern = black_box(HexPattern::new("", "", 256).unwrap());
    for counter in 0..6 {
        record_candidate(&mut h, rsa_modulus(&request, black_box(counter), &pattern));
    }
    h.finalize()
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
