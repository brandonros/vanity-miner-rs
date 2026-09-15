#[cfg(all(feature = "gpu", feature = "cumetal"))]
compile_error!("Select either gpu (NVIDIA CUDA) or cumetal, not both");

mod args;
mod common;
mod modes;
mod runner;

use crate::common::GlobalStats;
use args::Cli;
use clap::Parser;
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

    // Every search uses one reporter, including searches that have no matches yet.
    // Self-tests report their own assertions rather than search throughput.
    #[cfg(feature = "self_test_support")]
    if matches!(cli.command, args::Command::SelfTest) {
        return runner.run(&cli.command, stats);
    }
    let result = std::thread::scope(|scope| {
        let (done, finished) = std::sync::mpsc::channel::<()>();
        let observed = stats.clone();
        scope.spawn(move || {
            while matches!(
                finished.recv_timeout(std::time::Duration::from_secs(2)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                observed.print_progress();
            }
        });
        let result = runner.run(&cli.command, stats.clone());
        let _ = done.send(());
        result
    });
    stats.print_progress();
    result
}
