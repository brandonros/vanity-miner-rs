#![no_std]

extern crate alloc;
pub use llvm_metal_kernel;

pub mod crypto;
pub mod encoding;
pub mod modes;
pub mod search;

#[cfg(feature = "self_test_support")]
pub mod self_test;

#[cfg(feature = "test-vectors")]
pub mod test_vectors;
