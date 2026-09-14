mod args;
mod common;
#[cfg(feature = "crypto-cli")]
mod crypto_args;
#[cfg(feature = "crypto-cli")]
mod crypto_runner;
mod modes;
mod runner;

use crate::common::GlobalStats;
use args::Cli;
use clap::Parser;
use runner::Runner;
use std::error::Error;
use std::sync::Arc;

#[cfg(not(feature = "gpu"))]
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

    #[cfg(not(feature = "gpu"))]
    let runner = CpuRunner::new();

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

#[cfg(all(feature = "crypto-cli", feature = "gpu"))]
mod crypto_gpu;
