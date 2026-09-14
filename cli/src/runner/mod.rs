#[cfg(feature = "cumetal")]
mod cumetal;
#[cfg(feature = "cumetal")]
pub use cumetal::{CumetalOptions, CumetalRunner};

#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
mod cpu;
#[cfg(feature = "gpu")]
mod gpu;

#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
pub use cpu::CpuRunner;
#[cfg(feature = "gpu")]
pub use gpu::GpuRunner;

use crate::args::Command;
use crate::common::GlobalStats;
use std::error::Error;
use std::sync::Arc;

pub trait Runner {
    fn device_count(&self) -> usize;
    fn run(
        &self,
        command: &Command,
        stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
}

#[cfg(all(feature = "gpu", feature = "crypto-cli"))]
pub(crate) mod cuda_batches;
