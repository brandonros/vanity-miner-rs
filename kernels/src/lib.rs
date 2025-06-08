#![no_std]

extern crate alloc;

mod solana_vanity;
mod bitcoin_vanity;
mod shallenge;

pub use solana_vanity::*;
pub use bitcoin_vanity::*;
pub use shallenge::*;
