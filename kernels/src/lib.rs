//! # GPU kernel definitions
//!
//! ## Per-thread stack size
//!
//! Rust-CUDA's NVVM backend inlines whole pipelines per kernel. The k256
//! (secp256k1) derive — used by `ethereum` and `bitcoin` — pulls
//! in deep nested `.func` calls (`ProjectivePoint::add`/`::double`,
//! `FieldElement::invert`, plus the SecretKey→Scalar→AffinePoint chain) that
//! together overflow CUDA's default per-thread stack of **1024 bytes** the
//! moment they're composed with a runtime priv input. Launch faults with
//! `illegal memory access`; `compute-sanitizer` is the only tool that names
//! the cause directly.
//!
//! Empirically (RTX 4090, compute 8.9, CUDA 13.2): k256-derive kernels with
//! runtime priv need > 8 KiB and work at 16 KiB. Pipelines that don't compose
//! k256 (e.g. solana = xoroshiro + ed25519/dalek + base58) fit in the 1 KiB
//! default. The original 118-slot self-test ran clean at 16 KiB; RSA/P-256 checks use the larger host-configured stack.
//!
//! The CLI sets `cudaLimitStackSize = 16 KiB` by default in
//! `cli/src/runner/cuda/context.rs`; override with `STACK_SIZE=N`. If you add
//! a kernel that composes k256 with an even deeper consumer, bump that
//! default first.

#![no_std]

extern crate alloc;

#[cfg(all(target_arch = "nvptx64", feature = "self_test"))]
compile_error!(
    "GPU self-tests must be compiled separately with one self_test_<mode> feature; self_test is the host-test aggregate"
);

#[cfg(any(
    feature = "solana",
    feature = "bitcoin",
    feature = "ethereum",
    feature = "shallenge",
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-pss"
))]
mod match_handler;

#[cfg(feature = "bitcoin")]
pub mod bitcoin;
#[cfg(feature = "repro_nonce_sequence")]
pub mod codegen_repros;
#[cfg(feature = "ethereum")]
pub mod ethereum;
#[cfg(feature = "p256-public-key")]
pub mod p256_public_key;
#[cfg(feature = "p256-signature")]
pub mod p256_signature;
#[cfg(feature = "rsa-modulus")]
pub mod rsa_modulus;
#[cfg(feature = "rsa-pss")]
pub mod rsa_pss;
#[cfg(feature = "shallenge")]
pub mod shallenge;
#[cfg(feature = "solana")]
pub mod solana;

pub mod self_test;
