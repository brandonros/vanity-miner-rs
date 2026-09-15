#[cfg(all(
    not(any(feature = "gpu", feature = "cumetal")),
    any(
        feature = "solana",
        feature = "bitcoin",
        feature = "ethereum",
        feature = "shallenge"
    )
))]
mod cpu_workers;
#[cfg(feature = "gpu")]
mod gpu_context;
#[cfg(feature = "shallenge")]
mod shared_best_hash;
// The Runner interface retains legacy statistics even in new-mode-only builds.
#[cfg_attr(
    not(any(
        feature = "solana",
        feature = "bitcoin",
        feature = "ethereum",
        feature = "shallenge"
    )),
    allow(dead_code)
)]
mod stats;
#[cfg(any(
    feature = "solana",
    feature = "bitcoin",
    feature = "ethereum",
    feature = "shallenge"
))]
mod validation;

#[cfg(all(
    not(any(feature = "gpu", feature = "cumetal")),
    any(
        feature = "solana",
        feature = "bitcoin",
        feature = "ethereum",
        feature = "shallenge"
    )
))]
pub use cpu_workers::*;
#[cfg(feature = "gpu")]
pub use gpu_context::GpuContext;
#[cfg(feature = "shallenge")]
pub use shared_best_hash::*;
pub use stats::*;
#[cfg(any(
    feature = "solana",
    feature = "bitcoin",
    feature = "ethereum",
    feature = "shallenge"
))]
pub use validation::*;

#[cfg(all(feature = "crypto-cli", not(feature = "cumetal")))]
pub(crate) mod search_session;

#[cfg(feature = "gpu")]
pub(crate) mod cuda_module;
