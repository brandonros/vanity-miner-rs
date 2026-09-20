use super::args::EthereumArgs;
use crate::runner::{
    RunResult,
    metal::{MetalRunner, transport::Transport},
    progress::GlobalStats,
    session::run_device_session,
};
use std::sync::Arc;

pub fn run(runner: &MetalRunner, args: &EthereumArgs, stats: Arc<GlobalStats>) -> RunResult {
    let options = &runner.options;
    let mut engine = EthereumTransport::load(
        &runner.artifacts("ethereum"),
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
    engine.print_timings("Metal");
    result
}

#[path = "../../../../kernels/ethereum/src/contract.rs"]
mod contract;
pub type EthereumTransport = Transport<contract::Ethereum>;
