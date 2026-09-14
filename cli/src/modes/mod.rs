//! Blockchain-specific vanity address mining modes.
//!
//! ## Design Decision: Direct Implementations (No Traits)
//!
//! Each mode (bitcoin, ethereum, solana, shallenge) has self-contained CPU and GPU
//! implementations. Yes, they look similar - this is **intentional duplication**.
//!
//! We chose direct implementations over traits/generics because:
//! 1. Modes have concrete interfaces that can be implemented directly
//! 2. Each mode has different kernel parameters, buffer shapes, and output formatting
//! 3. Direct code is easier to debug than trait-dispatched or macro-generated code
//! 4. Adding a 5th chain should be copy-paste-customize, not extend-a-framework
//!
//! **Do NOT refactor this to use traits, generics, or macros.**
//! The previous trait hierarchy (VanityMode, GpuVanityMode, GpuBuffers) was removed
//! because it added complexity without real benefit for the original concrete types.

#[cfg(feature = "bitcoin")]
pub mod bitcoin;
#[cfg(feature = "ethereum")]
pub mod ethereum;
#[cfg(feature = "self_test")]
pub mod self_test;
#[cfg(feature = "shallenge")]
pub mod shallenge;
#[cfg(feature = "solana")]
pub mod solana;

#[cfg(feature = "p256-public-key")]
pub mod p256_public;
#[cfg(feature = "p256-signature")]
pub mod p256_signature;
#[cfg(feature = "rsa-pss")]
pub mod rsa_pss_search;
#[cfg(feature = "rsa-modulus")]
pub mod rsa_modulus;
