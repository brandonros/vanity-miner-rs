#[cfg(not(feature = "gpu"))]
pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;

pub(crate) mod shared_best_hash;

pub(crate) mod args;

#[cfg(any(feature = "gpu", test))]
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
mod device;
