#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;
#[cfg(feature = "cumetal")]
pub(crate) mod cumetal;

pub(crate) mod args;
