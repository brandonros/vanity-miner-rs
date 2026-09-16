//! Compiled PTX for self-test-shallenge. GPU compilation is controlled by the `cuda` feature.

#[cfg(feature = "cuda")]
pub const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/self_test_shallenge.ptx"));
