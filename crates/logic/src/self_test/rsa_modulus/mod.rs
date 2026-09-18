//! rsa modulus self-tests: primitives, resumable mining, and regressions.
mod fixtures;
pub(super) mod range_probes;
use super::known_answers::*;
use crate::self_test::black_box;
use fixtures::*;

register_self_test! {
    /// rsa modulus multiplication carry
    fn multiplication_carry() -> u32 {
        u32::from((|| {
            use crypto_bigint::{Encoding, U1024, U2048};
            let x = black_box(U1024::MAX);
            let result: U2048 = x.mul(&x);
            result.to_be_bytes() == CRYPTO_FIXTURE_MUL_CARRY
        })())
    }
}

register_self_test! {
    /// rsa modulus progression carry
    fn progression_carry() -> u32 {
        u32::from((|| {
            use crypto_bigint::{Encoding, U1024, U2048};
            let first = black_box(U1024::MAX);
            let stride = black_box(U1024::from_u8(2));
            let step: U2048 = stride.mul(&black_box(U1024::ONE));
            step.wrapping_add(&first.resize()).to_be_bytes() == CRYPTO_FIXTURE_PROGRESSION_CARRY
        })())
    }
}

register_self_test! {
    /// rsa modulus prime filter
    fn prime_filter() -> u32 {
        use crate::modes::rsa_modulus::probable_p;
        let mut even = black_box(SELF_TEST_RSA_P);
        even[127] &= 0xfe;
        u32::from(probable_p(&black_box(SELF_TEST_RSA_P)) && !probable_p(&even))
    }
}

register_self_test! {
    /// rsa modulus pseudoprime rejected
    fn pseudoprime_rejected() -> u32 {
        u32::from((|| {
            !crate::crypto::rsa_prime::probable_prime(&black_box(crypto_bigint::U1024::from_u64(
                341550071728321,
            )))
        })())
    }
}

register_self_test! {
    /// rsa modulus empty task rejected
    fn empty_task_rejected() -> u32 {
        use crate::modes::rsa_modulus::{self as pipeline, Task};
        u32::from(pipeline::q_at(&black_box(device_config()), &black_box(Task::EMPTY), 0).is_none())
    }
}

register_self_test! {
    /// rsa modulus upper bound rejected
    fn upper_bound_rejected() -> u32 {
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
}

register_self_test! {
    /// rsa modulus equal factors rejected
    fn equal_factors_rejected() -> u32 {
        use crate::modes::rsa_modulus as pipeline;
        let Ok(pattern) = crate::search::hex_pattern::HexPattern::new("", "", 256) else { return 0; };
        u32::from(!pipeline::eligible_pair(
            &black_box(SELF_TEST_RSA_P),
            &black_box(SELF_TEST_RSA_P),
            &pattern,
        ))
    }
}

register_self_test! {
    /// rsa modulus undersized factor rejected
    fn undersized_factor_rejected() -> u32 {
        use crate::modes::rsa_modulus as pipeline;
        let Ok(pattern) = crate::search::hex_pattern::HexPattern::new("", "", 256) else { return 0; };
        u32::from(!pipeline::eligible_pair(
            &black_box([0; 128]),
            &black_box(SELF_TEST_RSA_Q),
            &pattern,
        ))
    }
}

register_self_test! {
    /// end-to-end rsa modulus miner, including resume and factor retirement
    fn end_to_end() -> u32 {
        use crate::modes::rsa_modulus::{self as mining, Task};
        let config = black_box(device_config());
        let Ok(pattern) = crate::search::hex_pattern::HexPattern::new("", "", 256) else { return 0; };
        let mut task = Task::EMPTY;
        let (prepared, pair) = mining::mine(&config, &pattern, &mut task, black_box(9), 1, black_box(1));
        if pair.is_some() || prepared.errors != 0 || prepared.p_accepted != 1
            || prepared.ranges != 1 || prepared.q_tested != 0 || task.state != 2 {
            return 0;
        }
        let (searched, pair) = mining::mine(&config, &pattern, &mut task, black_box(10), 1, black_box(8));
        let Some(pair) = pair else { return 0; };
        u32::from(
            searched.errors == 0 && searched.p_tested == 0 && searched.q_tested == 1
                && searched.matches == 1 && pair.id == 9 && pair.p == SELF_TEST_RSA_P
                && pair.q == SELF_TEST_RSA_Q && task == Task::EMPTY
        )
    }
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

register_self_test! {
    /// rsa device full-width range construction
    fn device_range() -> u32 {
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
}

register_self_test! {
    /// rsa device range cursor wrap and retirement
    fn device_cursor() -> u32 {
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
}

register_self_test! {
    /// rsa device factor derivation known answer
    fn device_derivation() -> u32 {
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
}
