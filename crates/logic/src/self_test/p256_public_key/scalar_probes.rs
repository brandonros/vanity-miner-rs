//! Focused scalar probes for the p256_public_key self-test.
use crate::self_test::black_box;

register_self_test! {
    /// p256 order minus one produces negative generator
    fn order_minus_one() -> u32 {
        let mut scalar = super::CRYPTO_FIXTURE_P256_ORDER;
        scalar[31] -= 1;
        u32::from(
            crate::crypto::p256::public_point(&black_box(scalar))
                == Some(super::scalar_fixtures::NEGATIVE_GENERATOR),
        )
    }
}

register_self_test! {
    /// p256 order plus one rejected
    fn above_order() -> u32 {
        let mut scalar = super::CRYPTO_FIXTURE_P256_ORDER;
        scalar[31] += 1;
        u32::from(crate::crypto::p256::public_point(&black_box(scalar)).is_none())
    }
}
