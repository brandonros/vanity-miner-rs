#[cfg(feature = "cumetal")]
pub(crate) mod cumetal;
#[cfg(feature = "cumetal")]
pub use cumetal::{CumetalOptions, CumetalRunner};

#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;

#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
pub use cpu::CpuRunner;
#[cfg(feature = "gpu")]
pub use cuda::GpuRunner;

use crate::args::Command;
use crate::runner::progress::GlobalStats;
use std::error::Error;
use std::sync::Arc;

pub type RunResult = Result<(), Box<dyn Error + Send + Sync>>;

pub trait Runner {
    fn device_count(&self) -> usize;
    fn run(
        &self,
        command: &Command,
        stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
}

#[cfg(any(
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-modulus",
    feature = "rsa-pss"
))]
pub mod batches;
pub mod modules;
pub mod progress;
pub mod session;
pub mod workers;
