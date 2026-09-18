//! All production Metal and registry integration tests. Run scripts/test-gpu.sh.
#![cfg(all(feature = "metal", target_os = "macos"))]
#[path = "metal/bitcoin.rs"]
mod bitcoin;
#[path = "metal/ethereum.rs"]
mod ethereum;
#[path = "metal/p256_public.rs"]
mod p256_public;
#[path = "metal/p256_signature.rs"]
mod p256_signature;
#[path = "metal/rsa.rs"]
mod rsa;
#[path = "metal/rsa_pss.rs"]
mod rsa_pss;
#[path = "metal/self_test.rs"]
mod self_test;
#[path = "metal/shallenge.rs"]
mod shallenge;
#[path = "metal/solana.rs"]
mod solana;
#[path = "metal/support.rs"]
mod support;
