use crate::args::Cli;
use crate::runner::Runner;
use crate::runner::progress::GlobalStats;
use clap::Parser;
use std::error::Error;
use std::sync::Arc;

#[cfg(not(feature = "metal"))]
use crate::runner::CpuRunner;

pub fn run_cli() -> Result<(), Box<dyn Error + Send + Sync>> {
    let cli = Cli::parse();

    // Validate inputs
    cli.command.validate()?;

    #[cfg(feature = "self_test_support")]
    match &cli.command {
        crate::args::Command::SelfTest(args) if args.list => {
            for case in args.selected()? {
                println!("{}\t{}", case.name, case.label);
            }
            return Ok(());
        }
        #[allow(unreachable_patterns)]
        _ => {}
    }

    // Create runner based on compile-time feature
    #[cfg(not(feature = "metal"))]
    let runner = CpuRunner::new();

    #[cfg(feature = "metal")]
    let runner = crate::runner::metal::MetalRunner::new(cli.metal.clone())?;

    let details = cli.command.details();

    // Create stats
    let reporting_workers = runner.device_count();
    #[cfg(not(feature = "metal"))]
    let reporting_workers = details.cpu_threads.unwrap_or(reporting_workers);
    let stats = Arc::new(GlobalStats::new(
        reporting_workers,
        details.prefix_len,
        details.suffix_len,
    ));

    // Log what we're doing
    println!("{}", details.description);

    // Every search uses one reporter, including searches that have no matches yet.
    // Self-tests report their own assertions rather than search throughput.
    #[cfg(feature = "self_test_support")]
    if matches!(cli.command, crate::args::Command::SelfTest(_)) {
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
