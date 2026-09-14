//! Crypto used by the shared host/device logic.

#[cfg(feature = "solana")]
pub mod ed25519;
#[cfg(feature = "ethereum")]
pub mod keccak256;
#[cfg(any(feature = "p256-public-key", feature = "p256-signature"))]
pub mod p256_vanity;
#[cfg(feature = "bitcoin")]
pub mod ripemd160;
#[cfg(feature = "rsa-pss")]
pub mod rsa_crt;
#[cfg(feature = "rsa-modulus")]
pub mod rsa_prime;
#[cfg(feature = "rsa-pss")]
pub mod rsa_pss;
#[cfg(any(feature = "bitcoin", feature = "ethereum"))]
pub mod secp256k1;
#[cfg(any(feature = "bitcoin", feature = "shallenge"))]
pub mod sha256;
#[cfg(feature = "solana")]
pub mod sha512;
