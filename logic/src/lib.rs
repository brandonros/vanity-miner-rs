#![no_std]

extern crate alloc;

#[cfg(feature = "crypto-search")]
pub mod crypto_search;
#[cfg(feature = "crypto-search")]
pub mod candidate_result;
pub mod hex_pattern;
#[cfg(feature = "p256-public-key")]
pub mod p256_public_key_vanity;
#[cfg(feature = "p256-signature")]
pub mod p256_signature_vanity;
#[cfg(feature = "rsa-pss")]
pub mod rsa_pss_signature_vanity;
#[cfg(feature = "rsa-modulus")]
pub mod rsa_modulus_vanity;

#[cfg(any(feature = "p256-public-key", feature = "p256-signature"))]
pub mod p256_vanity;
#[cfg(feature = "rsa-pss")]
pub mod rsa_crt;
#[cfg(feature = "rsa-modulus")]
pub mod rsa_prime;
#[cfg(feature = "rsa-pss")]
pub mod rsa_pss;

#[cfg(any(feature = "solana", feature = "bitcoin"))]
mod base58;
#[cfg(feature = "bitcoin")]
mod bech32;
#[cfg(feature = "bitcoin")]
mod bitcoin_vanity;
#[cfg(feature = "solana")]
mod ed25519;
#[cfg(feature = "ethereum")]
mod ethereum_vanity;
#[cfg(feature = "ethereum")]
mod keccak256;
#[cfg(feature = "bitcoin")]
mod ripemd160;
#[cfg(any(feature = "bitcoin", feature = "ethereum"))]
mod secp256k1;
#[cfg(feature = "self_test")]
mod self_test;
#[cfg(any(feature = "bitcoin", feature = "shallenge"))]
mod sha256;
#[cfg(feature = "solana")]
mod sha512;
#[cfg(feature = "shallenge")]
mod shallenge;
#[cfg(feature = "solana")]
mod solana_vanity;
#[cfg(any(feature = "solana", feature = "bitcoin", feature = "ethereum"))]
mod vanity;
mod xoroshiro;

#[cfg(any(feature = "solana", feature = "bitcoin"))]
pub use base58::*;
#[cfg(feature = "bitcoin")]
pub use bech32::*;
#[cfg(feature = "bitcoin")]
pub use bitcoin_vanity::*;
#[cfg(feature = "solana")]
pub use ed25519::*;
#[cfg(feature = "ethereum")]
pub use ethereum_vanity::*;
#[cfg(feature = "ethereum")]
pub use keccak256::*;
#[cfg(feature = "bitcoin")]
pub use ripemd160::*;
#[cfg(any(feature = "bitcoin", feature = "ethereum"))]
pub use secp256k1::*;
#[cfg(feature = "self_test")]
pub use self_test::*;
#[cfg(any(feature = "bitcoin", feature = "shallenge"))]
pub use sha256::*;
#[cfg(feature = "solana")]
pub use sha512::*;
#[cfg(feature = "shallenge")]
pub use shallenge::*;
#[cfg(feature = "solana")]
pub use solana_vanity::*;
pub use xoroshiro::*;
