#[cfg(not(any(feature = "gpu", feature = "cumetal", feature = "metal")))]
pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;
#[cfg(feature = "cumetal")]
pub(crate) mod cumetal;
#[cfg(feature = "metal")]
pub(crate) mod metal;

pub(crate) mod args;

#[cfg(any(feature = "gpu", feature = "cumetal", feature = "metal", test))]
#[cfg_attr(
    not(any(feature = "gpu", feature = "cumetal", feature = "metal")),
    allow(dead_code)
)]
mod device;
