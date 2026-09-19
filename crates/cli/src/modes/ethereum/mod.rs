#[cfg(not(feature = "metal"))]
pub(crate) mod cpu;
#[cfg(feature = "metal")]
pub mod metal;

pub(crate) mod args;

#[cfg(any(feature = "metal", test))]
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
mod device;
