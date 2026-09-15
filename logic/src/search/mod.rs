//! Search used by the shared host/device logic.

#[cfg(any(
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-pss",
    feature = "rsa-modulus"
))]
pub mod candidate_result;
#[cfg(any(
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-pss",
    feature = "rsa-modulus"
))]
pub mod crypto_search;
pub mod hex_pattern;
#[cfg(any(feature = "solana", feature = "bitcoin", feature = "ethereum"))]
pub mod vanity;
pub mod xoroshiro;
