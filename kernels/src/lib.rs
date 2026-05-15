//! # GPU kernel definitions
//!
//! ## Per-thread stack size
//!
//! Rust-CUDA's NVVM backend inlines whole pipelines per kernel. The k256
//! (secp256k1) derive — used by `ethereum_vanity` and `bitcoin_vanity` — pulls
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
//! default. The full 118-slot self-test runs clean at 16 KiB.
//!
//! The CLI sets `cudaLimitStackSize = 16 KiB` by default in
//! `cli/src/common/gpu_context.rs`; override with `STACK_SIZE=N`. If you add
//! a kernel that composes k256 with an even deeper consumer, bump that
//! default first.

#![no_std]

extern crate alloc;

mod atomic;
#[macro_use]
mod match_handler;
mod utilities;

mod solana_vanity;
mod bitcoin_vanity;
mod ethereum_vanity;
mod shallenge;
mod self_test;

pub use solana_vanity::*;
pub use bitcoin_vanity::*;
pub use ethereum_vanity::*;
pub use shallenge::*;
pub use self_test::*;
