//! Stateless range construction and sampling probes.
use crate::modes::rsa_modulus as pipeline;
use core::hint::black_box;
use crypto_bigint::{Encoding, U1024, U2048};

register_self_test! {
    /// rsa empty interval has no candidate
    fn range_empty() -> u32 {
        let mut config = super::device_config();
        let n = U2048::from_be_slice(&config.lower)
            .wrapping_add(&U2048::ONE)
            .to_be_bytes();
        config.lower = n;
        config.upper = n;
        config.suffix_bits = 1;
        u32::from(
            pipeline::generate_q(&black_box(config), &black_box(super::SELF_TEST_RSA_P), 9) == Ok(None),
        )
    }
}

register_self_test! {
    /// rsa five-value range and independent HMAC known answer
    fn range_multiple() -> u32 {
        let mut config = super::device_config();
        let p = U1024::from_be_slice(&super::SELF_TEST_RSA_P);
        let mut last = super::SELF_TEST_RSA_Q;
        last[127] += 8;
        config.upper = p.mul(&U1024::from_be_slice(&last)).to_be_bytes();
        config.suffix_bits = 1;
        // Independent Python HMAC oracle: this ID samples index zero of five.
        u32::from(
            pipeline::progression(&black_box(config), &black_box(super::SELF_TEST_RSA_P))
                == Some((super::SELF_TEST_RSA_Q, U1024::from_u8(5).to_be_bytes()))
                && pipeline::generate_q(&config, &super::SELF_TEST_RSA_P, black_box(9))
                    == Ok(Some(super::SELF_TEST_RSA_Q)),
        )
    }
}

register_self_test! {
    /// rsa single-value range remains independently sampleable
    fn range_single_value() -> u32 {
        let config = black_box(super::device_config());
        for id in [0, 1, u64::MAX] {
            if pipeline::generate_q(&config, &black_box(super::SELF_TEST_RSA_P), black_box(id))
                != Ok(Some(super::SELF_TEST_RSA_Q))
            {
                return 0;
            }
        }
        1
    }
}

register_self_test! {
    /// rsa range containing only inseparable factors is empty
    fn range_separation() -> u32 {
        let mut config = super::device_config();
        let p = U1024::from_be_slice(&super::SELF_TEST_RSA_P);
        let n: U2048 = p.mul(&p);
        config.lower = n.to_be_bytes();
        config.upper = config.lower;
        config.suffix = config.lower;
        u32::from(
            pipeline::generate_q(&black_box(config), &black_box(super::SELF_TEST_RSA_P), 9) == Ok(None),
        )
    }
}
