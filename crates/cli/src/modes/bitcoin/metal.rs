use super::args::BitcoinArgs;
use crate::runner::{
    RunResult,
    metal::{MetalRunner, unified::Transport},
    progress::GlobalStats,
    session::run_device_session,
};
use std::sync::Arc;

pub fn run(runner: &MetalRunner, args: &BitcoinArgs, stats: Arc<GlobalStats>) -> RunResult {
    let options = &runner.options;
    let mut engine = BitcoinTransport::load(
        &runner.artifacts("bitcoin"),
        options.batch_size,
        options.threads_per_group as usize,
        options.verify,
    )?;
    let result = run_device_session(
        stats,
        "keys",
        options.batches,
        options.batch_size,
        runner.exit_on_first_match,
        |control| {
            super::device::search(
                &args.prefix,
                &args.suffix,
                options.seed,
                &control,
                |seed, pattern, _, start, count| engine.evaluate(seed, pattern, &[], start, count),
            )
            .map(|_| ())
        },
    );
    eprintln!(
        "Metal: {} launches; library {:.3} ms; pipeline {:.3} ms; dispatch {:.3} ms; GPU {:.3} ms ({} timed); validation {:.3} ms",
        engine.launches,
        engine.load_stages.library.as_secs_f64() * 1000.,
        engine.load_stages.pipeline.as_secs_f64() * 1000.,
        engine.dispatch_time.as_secs_f64() * 1000.,
        engine.gpu_time.as_secs_f64() * 1000.,
        engine.gpu_timed_launches,
        engine.verification_time.as_secs_f64() * 1000.,
    );
    result
}

#[path = "../../../../kernels/bitcoin/src/contract.rs"]
mod contract;
pub type BitcoinTransport = Transport<contract::Bitcoin>;
