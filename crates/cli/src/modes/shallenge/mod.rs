#[cfg(not(feature = "metal"))]
pub(crate) mod cpu;

pub(crate) mod shared_best_hash;

pub(crate) mod args;

#[cfg(any(feature = "metal", test))]
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
mod device;

#[cfg(feature = "metal")]
pub mod metal;
