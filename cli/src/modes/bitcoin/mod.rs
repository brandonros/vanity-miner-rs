#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;
#[cfg(feature = "cumetal")]
pub(crate) mod cumetal;

pub(crate) mod args;

#[cfg(any(feature = "gpu", feature = "cumetal", test))]
#[cfg_attr(not(any(feature = "gpu", feature = "cumetal")), allow(dead_code))]
mod device;
