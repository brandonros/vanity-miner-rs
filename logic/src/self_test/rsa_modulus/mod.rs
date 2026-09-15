//! rsa modulus self-tests: primitives, pipeline stages, and regressions.
mod fixtures;
use super::known_answers::*;
use core::hint::black_box;
use fixtures::*;

#[inline(never)]
pub fn check_rsa_modulus_multiplication_carry() -> u32 {
    u32::from((|| {
        use crypto_bigint::{Encoding, U1024, U2048};
        let x = black_box(U1024::MAX);
        let result: U2048 = x.mul(&x);
        result.to_be_bytes() == CRYPTO_FIXTURE_MUL_CARRY
    })())
}

#[inline(never)]
pub fn check_rsa_modulus_progression_carry() -> u32 {
    u32::from((|| {
        use crypto_bigint::{Encoding, U1024, U2048};
        let first = black_box(U1024::MAX);
        let stride = black_box(U1024::from_u8(2));
        let step: U2048 = stride.mul(&black_box(U1024::ONE));
        step.wrapping_add(&first.resize()).to_be_bytes() == CRYPTO_FIXTURE_PROGRESSION_CARRY
    })())
}

#[inline(never)]
pub fn check_rsa_modulus_prime_filter() -> u32 {
    use crate::modes::rsa_modulus::probable_p;
    let mut even = black_box(SELF_TEST_RSA_P);
    even[127] &= 0xfe;
    u32::from(probable_p(&black_box(SELF_TEST_RSA_P)) && !probable_p(&even))
}

#[inline(never)]
pub fn check_rsa_modulus_pseudoprime_rejected() -> u32 {
    u32::from((|| {
        !crate::crypto::rsa_prime::probable_prime(&black_box(crypto_bigint::U1024::from_u64(
            341550071728321,
        )))
    })())
}

#[inline(never)]
pub fn check_rsa_modulus_empty_task_rejected() -> u32 {
    use crate::modes::rsa_modulus::{self as pipeline, Task};
    u32::from(pipeline::q_at(&black_box(device_config()), &black_box(Task::EMPTY), 0).is_none())
}

#[inline(never)]
pub fn check_rsa_modulus_upper_bound_rejected() -> u32 {
    use crate::modes::rsa_modulus::{self as pipeline, Task};
    let config = black_box(device_config());
    let mut task = black_box(Task {
        p: SELF_TEST_RSA_P,
        state: 1,
        ..Task::EMPTY
    });
    u32::from(
        pipeline::prepare_range(&config, &mut task) == Ok(true)
            && pipeline::q_at(&config, &task, 1).is_none(),
    )
}

#[inline(never)]
pub fn check_rsa_modulus_equal_factors_rejected() -> u32 {
    use crate::modes::rsa_modulus as pipeline;
    let pattern = crate::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
    u32::from(!pipeline::eligible_pair(
        &black_box(SELF_TEST_RSA_P),
        &black_box(SELF_TEST_RSA_P),
        &pattern,
    ))
}

#[inline(never)]
pub fn check_rsa_modulus_undersized_factor_rejected() -> u32 {
    use crate::modes::rsa_modulus as pipeline;
    let pattern = crate::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
    u32::from(!pipeline::eligible_pair(
        &black_box([0; 128]),
        &black_box(SELF_TEST_RSA_Q),
        &pattern,
    ))
}

#[inline(never)]
pub fn check_rsa_modulus_end_to_end() -> u32 {
    use crate::modes::rsa_modulus::{self as pipeline, Task};
    let config = black_box(device_config());
    let pattern = crate::search::hex_pattern::HexPattern::new("", "", 256).unwrap();
    let Some(p) = pipeline::generate_p(&config, black_box(9)) else {
        return 0;
    };
    if p != SELF_TEST_RSA_P || !pipeline::probable_p(&p) {
        return 0;
    }
    let mut task = Task {
        p,
        id: 9,
        state: 1,
        ..Task::EMPTY
    };
    if pipeline::prepare_range(&config, &mut task) != Ok(true) {
        return 0;
    }
    let Some(q) = pipeline::q_at(&config, &task, black_box(0)) else {
        return 0;
    };
    if q != SELF_TEST_RSA_Q || !pipeline::eligible_pair(&p, &q, &pattern) {
        return 0;
    }
    task.winner = 1;
    pipeline::finish_tile(&mut task, 1);
    u32::from(task.state == 0 && task.p == [0; 128])
}

/// Write only this mode's stable result slots.
pub fn run(results: &mut [u32]) {
    results[145] = check_rsa_modulus_multiplication_carry();
    results[146] = check_rsa_modulus_progression_carry();
    results[147] = check_rsa_modulus_prime_filter();
    results[148] = check_rsa_modulus_pseudoprime_rejected();
    results[149] = check_rsa_modulus_empty_task_rejected();
    results[150] = check_rsa_modulus_upper_bound_rejected();
    results[151] = check_rsa_modulus_equal_factors_rejected();
    results[152] = check_rsa_modulus_undersized_factor_rejected();
    results[156] = check_rsa_modulus_end_to_end();
    results[157] = check_device_range();
    results[158] = check_device_cursor();
    results[159] = check_device_derivation();
}

fn device_config() -> crate::modes::rsa_modulus::SearchConfig {
    use crypto_bigint::{Encoding, U1024, U2048};
    let n: U2048 =
        U1024::from_be_slice(&SELF_TEST_RSA_P).mul(&U1024::from_be_slice(&SELF_TEST_RSA_Q));
    crate::modes::rsa_modulus::SearchConfig {
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

#[inline(never)]
pub fn check_device_range() -> u32 {
    use crate::modes::rsa_modulus::{self as pipeline, Task};
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

#[inline(never)]
pub fn check_device_cursor() -> u32 {
    use crate::modes::rsa_modulus::{self as pipeline, Task};
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

#[inline(never)]
pub fn check_device_derivation() -> u32 {
    use crate::modes::rsa_modulus as pipeline;
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
