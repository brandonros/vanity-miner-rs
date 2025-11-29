//! Blockchain-specific vanity address mining modes.
//!
//! ## Design Decision: Direct Implementations (No Traits)
//!
//! Each mode (bitcoin, ethereum, solana, shallenge) has self-contained CPU and GPU
//! implementations. Yes, they look similar - this is **intentional duplication**.
//!
//! We chose direct implementations over traits/generics because:
//! 1. Only 4 modes exist - below the threshold where abstraction pays off
//! 2. Each mode has different kernel parameters, buffer shapes, and output formatting
//! 3. Direct code is easier to debug than trait-dispatched or macro-generated code
//! 4. Adding a 5th chain should be copy-paste-customize, not extend-a-framework
//!
//! **Do NOT refactor this to use traits, generics, or macros.**
//! The previous trait hierarchy (VanityMode, GpuVanityMode, GpuBuffers) was removed
//! because it added complexity without real benefit for 4 concrete types.

pub mod bitcoin;
pub mod ethereum;
pub mod shallenge;
pub mod solana;
