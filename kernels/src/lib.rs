#![no_std]

extern crate alloc;

mod atomic;
#[macro_use]
mod match_handler;
mod memory;
mod panic;
mod utilities;

mod solana_vanity;
mod bitcoin_vanity;
mod ethereum_vanity;
mod shallenge;

pub use solana_vanity::*;
pub use bitcoin_vanity::*;
pub use ethereum_vanity::*;
pub use shallenge::*;
