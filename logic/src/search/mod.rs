//! Search used by the shared host/device logic.

#[cfg(any(
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-modulus"
))]
pub mod candidate_derivation;
#[cfg(any(
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-pss",
    feature = "rsa-modulus"
))]
pub mod candidate_result;
pub mod hex_pattern;
#[cfg(any(feature = "p256-signature", feature = "rsa-pss"))]
pub mod message_window;
#[cfg(feature = "rsa-pss")]
pub mod salt_counter;
#[cfg(any(feature = "solana", feature = "bitcoin", feature = "ethereum"))]
pub mod vanity;
pub mod xoroshiro;

pub mod device_record;
