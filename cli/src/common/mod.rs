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
#[cfg(any(
    feature = "solana",
    feature = "bitcoin",
    feature = "ethereum",
    feature = "shallenge"
))]
pub use validation::*;
pub use vanity_miner::stats::*;

#[cfg(feature = "crypto-cli")]
pub(crate) mod search_session;

#[cfg(feature = "gpu")]
pub(crate) mod cuda_module;

#[cfg(feature = "crypto-cli")]
pub(crate) mod pattern_args;
