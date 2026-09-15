//! rsa modulus self-tests: primitives, pipeline stages, and regressions.
mod fixtures;
use super::known_answers::*;
use super::record_candidate;
use core::hint::black_box;
use fixtures::*;

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
        use crate::modes::rsa_modulus::{RsaModulusRequest, rsa_modulus};
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
        use crate::modes::rsa_modulus::{RsaModulusRequest, rsa_modulus};
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
        use crate::modes::rsa_modulus::{RsaModulusRequest, rsa_modulus};
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
        use crate::modes::rsa_modulus::{RsaModulusRequest, rsa_modulus};
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
        use crate::modes::rsa_modulus::{RsaModulusRequest, rsa_modulus};
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
    use crate::{modes::rsa_modulus::*, search::hex_pattern::HexPattern};
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

/// Write only this mode's stable result slots.
pub fn run(results: &mut [u32]) {
    results[145] = check_rsa_modulus_multiplication_carry();
    results[146] = check_rsa_modulus_progression_carry();
    results[147] = check_rsa_modulus_prime_filter();
    results[148] = check_rsa_modulus_pseudoprime_rejected();
    results[149] = check_rsa_modulus_zero_stride_rejected();
    results[150] = check_rsa_modulus_upper_bound_rejected();
    results[151] = check_rsa_modulus_equal_factors_rejected();
    results[152] = check_rsa_modulus_undersized_factor_rejected();
    results[156] = check_rsa_modulus_end_to_end();
    results[157] = check_device_range();
    results[158] = check_device_cursor();
    results[159] = check_device_derivation();
}

fn device_config() -> crate::modes::rsa_modulus::pipeline::SearchConfig {
    use crypto_bigint::{Encoding, U1024, U2048};
    let n: U2048 =
        U1024::from_be_slice(&SELF_TEST_RSA_P).mul(&U1024::from_be_slice(&SELF_TEST_RSA_Q));
    crate::modes::rsa_modulus::pipeline::SearchConfig {
        lower: n.to_be_bytes(),
        upper: n.to_be_bytes(),
        suffix: n.to_be_bytes(),
        p_min: SELF_TEST_RSA_P,
        p_count: U1024::ONE.to_be_bytes(),
        seed: [42; 32],
        worker: 7,
        suffix_bits: 2048,
        reserved: 0,
    }
}

pub fn check_device_range() -> u32 {
    use crate::modes::rsa_modulus::pipeline::{self, Task};
    let config = black_box(device_config());
    let mut task = black_box(Task {
        p: SELF_TEST_RSA_P,
        state: 1,
        id: 9,
        ..Task::EMPTY
    });
    u32::from(
        pipeline::prepare_range(&config, &mut task) == Ok(true)
            && pipeline::q_at(&config, &task, black_box(0)) == Some(SELF_TEST_RSA_Q)
            && pipeline::q_at(&config, &task, black_box(1)).is_none(),
    )
}

pub fn check_device_cursor() -> u32 {
    use crate::modes::rsa_modulus::pipeline::{self, Task};
    use crypto_bigint::{Encoding, U1024};
    let mut config = black_box(device_config());
    config.suffix_bits = 1;
    let first = black_box(U1024::ONE.shl_vartime(1023).wrapping_add(&U1024::ONE));
    let mut task = black_box(Task {
        state: 2,
        first: first.to_be_bytes(),
        count: U1024::from_u32(5).to_be_bytes(),
        cursor: U1024::from_u32(4).to_be_bytes(),
        remaining: U1024::from_u32(5).to_be_bytes(),
        ..Task::EMPTY
    });
    for (offset, delta) in [8, 0, 2, 4, 6].into_iter().enumerate() {
        if pipeline::q_at(&config, &task, black_box(offset as u32))
            != Some(first.wrapping_add(&U1024::from_u32(delta)).to_be_bytes())
        {
            return 0;
        }
    }
    pipeline::finish_tile(&mut task, black_box(2));
    if task.remaining != U1024::from_u32(3).to_be_bytes() || task.cursor != U1024::ONE.to_be_bytes()
    {
        return 0;
    }
    task.winner = black_box(1);
    pipeline::finish_tile(&mut task, black_box(3));
    u32::from(task.state == 0 && task.p == [0; 128] && task.remaining == [0; 128])
}

pub fn check_device_derivation() -> u32 {
    use crate::modes::rsa_modulus::pipeline;
    use crypto_bigint::{Encoding, U1024};
    let mut config = black_box(device_config());
    config.p_min = U1024::ONE
        .shl_vartime(1023)
        .wrapping_add(&U1024::ONE)
        .to_be_bytes();
    config.p_count = U1024::ONE.shl_vartime(1022).to_be_bytes();
    // Independently computed using Python hmac/hashlib, including the rejection mask.
    let expected = U1024::from_be_hex(
        "91997e74ee42e6aa8794846bd8857a1eb01fffb3934228b1d30c55d05a6c571875375e6da80876ee34d0af5a52b2fc5447feba392ab0186ab33970e59aeb45e2d4400d49e01a05c948936982d9f2289e22d5aa7a90c6a8b18e330239d6338f650a74aa3baaa510fcbb6050e6cebcd054d3aacfb7e0e8de73c8fca280ee840019",
    );
    u32::from(pipeline::generate_p(&config, black_box(9)) == Some(expected.to_be_bytes()))
}
