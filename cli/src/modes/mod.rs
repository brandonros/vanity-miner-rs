//! Concrete mode entry points. Each mode owns its arguments and backend adapters.
//! Keep algorithm and output choices here; share only transport and coordination.
//! Do not introduce a mode trait hierarchy or mode-generating macros.

#[cfg(feature = "bitcoin")]
pub mod bitcoin;
#[cfg(feature = "ethereum")]
pub mod ethereum;
#[cfg(all(feature = "self_test_support", not(feature = "cumetal")))]
pub mod self_test;
#[cfg(feature = "shallenge")]
pub mod shallenge;
#[cfg(feature = "solana")]
pub mod solana;

#[cfg(feature = "p256-public-key")]
pub mod p256_public_key;
#[cfg(feature = "p256-signature")]
pub mod p256_signature;
#[cfg(feature = "rsa-modulus")]
pub mod rsa_modulus;
#[cfg(feature = "rsa-pss")]
pub mod rsa_pss;
