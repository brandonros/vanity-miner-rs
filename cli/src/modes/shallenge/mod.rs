#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;
#[cfg(feature = "cumetal")]
pub(crate) mod cumetal;

#[cfg(not(feature = "cumetal"))]
pub(crate) mod shared_best_hash;

pub(crate) mod args;
