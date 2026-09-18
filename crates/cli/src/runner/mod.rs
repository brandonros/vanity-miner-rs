#[cfg(not(feature = "metal"))]
mod cpu;

#[cfg(not(feature = "metal"))]
pub use cpu::CpuRunner;

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

pub mod batches;
pub mod progress;
pub mod session;
pub mod workers;

#[cfg(feature = "metal")]
pub mod metal;
