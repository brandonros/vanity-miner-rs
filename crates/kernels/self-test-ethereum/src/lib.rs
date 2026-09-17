//! Compiled PTX for self-test-ethereum. GPU compilation is controlled by the `cuda` feature.

#[cfg(feature = "cuda")]
pub const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/self_test_ethereum.ptx"));
