//! Host searches, grouped by the mode that owns their keys and candidates.
#[cfg(feature = "p256-public-key")]
pub mod p256_public_key;
#[cfg(feature = "p256-signature")]
pub mod p256_signature;
#[cfg(feature = "rsa-modulus")]
pub mod rsa_modulus;
#[cfg(feature = "rsa-pss")]
pub mod rsa_pss;

#[cfg(any(
    feature = "rsa-modulus",
    feature = "rsa-pss",
    feature = "p256-public-key",
    feature = "p256-signature"
))]
mod worker_results;
