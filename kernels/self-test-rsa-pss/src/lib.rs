//! Compiled PTX for self-test-rsa-pss. GPU compilation is controlled by the `cuda` feature.

#[cfg(feature = "cuda")]
pub const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/self_test_rsa_pss.ptx"));
