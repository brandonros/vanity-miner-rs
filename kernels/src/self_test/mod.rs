//! On-device known-answer entry points, one module per enabled feature.
#[cfg(feature = "self_test_bitcoin")]
pub mod bitcoin;
#[cfg(feature = "self_test_ethereum")]
pub mod ethereum;
#[cfg(feature = "self_test_p256_public_key")]
pub mod p256_public_key;
#[cfg(feature = "self_test_p256_signature")]
pub mod p256_signature;
#[cfg(feature = "self_test_rsa_modulus")]
pub mod rsa_modulus;
#[cfg(feature = "self_test_rsa_pss")]
pub mod rsa_pss;
#[cfg(feature = "self_test_shallenge")]
pub mod shallenge;
#[cfg(feature = "self_test_solana")]
pub mod solana;
