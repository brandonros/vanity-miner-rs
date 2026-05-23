#![no_std]

extern crate alloc;

mod xoroshiro;
#[cfg(any(feature = "solana", feature = "bitcoin"))]
mod base58;
#[cfg(feature = "bitcoin")]
mod bech32;
#[cfg(feature = "solana")]
mod ed25519;
#[cfg(any(feature = "bitcoin", feature = "ethereum"))]
mod secp256k1;
#[cfg(any(feature = "bitcoin", feature = "shallenge"))]
mod sha256;
#[cfg(feature = "solana")]
mod sha512;
#[cfg(feature = "bitcoin")]
mod ripemd160;
#[cfg(feature = "shallenge")]
mod shallenge;
#[cfg(feature = "ethereum")]
mod keccak256;
#[cfg(any(feature = "solana", feature = "bitcoin", feature = "ethereum"))]
mod vanity;
#[cfg(feature = "solana")]
mod solana_vanity;
#[cfg(feature = "bitcoin")]
mod bitcoin_vanity;
#[cfg(feature = "ethereum")]
mod ethereum_vanity;
#[cfg(feature = "self_test")]
mod self_test;

pub use xoroshiro::*;
#[cfg(any(feature = "solana", feature = "bitcoin"))]
pub use base58::*;
#[cfg(feature = "bitcoin")]
pub use bech32::*;
#[cfg(feature = "solana")]
pub use ed25519::*;
#[cfg(any(feature = "bitcoin", feature = "ethereum"))]
pub use secp256k1::*;
#[cfg(any(feature = "bitcoin", feature = "shallenge"))]
pub use sha256::*;
#[cfg(feature = "solana")]
pub use sha512::*;
#[cfg(feature = "bitcoin")]
pub use ripemd160::*;
#[cfg(feature = "shallenge")]
pub use shallenge::*;
#[cfg(feature = "ethereum")]
pub use keccak256::*;
#[cfg(feature = "solana")]
pub use solana_vanity::*;
#[cfg(feature = "bitcoin")]
pub use bitcoin_vanity::*;
#[cfg(feature = "ethereum")]
pub use ethereum_vanity::*;
#[cfg(feature = "self_test")]
pub use self_test::*;
