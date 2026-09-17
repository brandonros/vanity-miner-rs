//! Compiled PTX for rsa-pss. GPU compilation is controlled by the `cuda` feature.

#[cfg(feature = "cuda")]
pub const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/rsa_pss.ptx"));
