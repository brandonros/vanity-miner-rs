//! Modes used by the shared host/device logic.

#[cfg(feature = "bitcoin")]
pub mod bitcoin;
#[cfg(feature = "ethereum")]
pub mod ethereum;
#[cfg(feature = "p256-public-key")]
pub mod p256_public_key;
#[cfg(feature = "p256-signature")]
pub mod p256_signature;
#[cfg(feature = "rsa-pss")]
pub mod rsa_pss;
#[cfg(feature = "shallenge")]
pub mod shallenge;
#[cfg(feature = "solana")]
pub mod solana;
