//! Compiled PTX for self-test-p256-public-key. GPU compilation is controlled by the `cuda` feature.

#[cfg(feature = "cuda")]
pub const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/self_test_p256_public_key.ptx"));
