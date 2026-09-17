//! Encoding used by the shared host/device logic.

#[cfg(any(feature = "solana", feature = "bitcoin"))]
pub mod base58;
#[cfg(feature = "bitcoin")]
pub mod bech32;
