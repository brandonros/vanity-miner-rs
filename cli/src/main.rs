mod args;
mod common;
mod modes;
mod runner;

use args::Cli;
use clap::Parser;
use crate::common::GlobalStats;
use runner::{CpuRunner, Runner};
use std::error::Error;
use std::sync::Arc;

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
