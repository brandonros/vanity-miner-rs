//! rsa modulus self-tests: primitives, independent candidates, and regressions.
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
    /// rsa modulus zero factor count rejected
    fn zero_count_rejected() -> u32 {
        let mut config = black_box(device_config());
        config.p_count = [0; 128];
        let Ok(pattern) = crate::search::hex_pattern::HexPattern::new("", "", 256) else { return 0; };
        u32::from(crate::modes::rsa_modulus::rsa_modulus(&config, 9, &pattern).status == 2)
    }
}

register_self_test! {
    /// rsa modulus inverted interval rejected
    fn upper_bound_rejected() -> u32 {
        let mut config = black_box(device_config());
        config.upper = [0; 256];
        u32::from(
            crate::modes::rsa_modulus::generate_q(&config, &black_box(SELF_TEST_RSA_P), 9) == Ok(None),
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
    /// end-to-end independent rsa modulus candidate
    fn end_to_end() -> u32 {
        let config = black_box(device_config());
        let Ok(pattern) = crate::search::hex_pattern::HexPattern::new("", "", 256) else { return 0; };
        let result = crate::modes::rsa_modulus::rsa_modulus(&config, black_box(9), &pattern);
        u32::from(
            result.status == 1
                && result.bytes[..128] == SELF_TEST_RSA_P
                && result.bytes[128..] == SELF_TEST_RSA_Q,
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
    /// rsa device full-width range sampling
    fn device_range() -> u32 {
        u32::from(
            crate::modes::rsa_modulus::generate_q(
                &black_box(device_config()),
                &black_box(SELF_TEST_RSA_P),
                black_box(9),
            ) == Ok(Some(SELF_TEST_RSA_Q)),
        )
    }
}

register_self_test! {
    /// rsa candidate evaluation is independent of intervening IDs
    fn candidate_repeatability() -> u32 {
        let config = black_box(device_config());
        let p = black_box(SELF_TEST_RSA_P);
        let first = crate::modes::rsa_modulus::generate_q(&config, &p, black_box(9));
        let _ = black_box(crate::modes::rsa_modulus::generate_q(
            &config,
            &p,
            black_box(123),
        ));
        u32::from(
            first == Ok(Some(SELF_TEST_RSA_Q))
                && first == crate::modes::rsa_modulus::generate_q(&config, &p, black_box(9)),
        )
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
