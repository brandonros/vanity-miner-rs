#[cfg(all(feature = "gpu", feature = "cumetal"))]
compile_error!("Select either gpu (NVIDIA CUDA) or cumetal, not both");

mod args;
mod common;
#[cfg(not(feature = "cumetal"))]
mod modes;
mod runner;

use args::Cli;
use clap::Parser;
use crate::common::GlobalStats;
use runner::Runner;
use std::error::Error;
use std::sync::Arc;

#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
use runner::CpuRunner;

#[cfg(feature = "gpu")]
use runner::GpuRunner;

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let cli = Cli::parse();

    // Validate inputs
    cli.command.validate()?;

    // Create runner based on compile-time feature
    #[cfg(feature = "gpu")]
    let runner = GpuRunner::new()?;

    #[cfg(not(any(feature = "gpu", feature = "cumetal")))]
    let runner = CpuRunner::new();

    #[cfg(feature = "cumetal")]
    let runner = runner::CumetalRunner::new(cli.cumetal.clone())?;

    // Create stats
    let stats = Arc::new(GlobalStats::new(
        runner.device_count(),
        cli.command.prefix_len(),
        cli.command.suffix_len(),
    ));

    // Log what we're doing
    println!("{}", cli.command.description());

    // Run
    runner.run(&cli.command, stats)
}
