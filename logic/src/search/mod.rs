//! Search used by the shared host/device logic.

#[cfg(feature = "crypto-search")]
pub mod candidate_result;
#[cfg(feature = "crypto-search")]
pub mod crypto_search;
pub mod hex_pattern;
#[cfg(any(feature = "solana", feature = "bitcoin", feature = "ethereum"))]
pub mod vanity;
pub mod xoroshiro;
