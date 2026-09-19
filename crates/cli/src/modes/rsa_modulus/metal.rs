use crate::runner::{
    RunResult,
    metal::{MetalRunner, transport::Transport},
    progress::GlobalStats,
    session::run_device_session,
};
use std::sync::Arc;
pub fn run(
    runner: &MetalRunner,
    args: &super::args::RsaModulusArgs,
    stats: Arc<GlobalStats>,
) -> RunResult {
    let options = &runner.options;
    if options.seed.is_some() {
        return Err("RSA uses OS cryptographic entropy; --seed is not supported".into());
    }
    let config = args.config(1)?;
    let mut engine = RsaTransport::load(
        &runner.artifacts("rsa-modulus"),
        options.batch_size,
        options.threads_per_group as usize,
        options.verify,
    )?;
    let result = run_device_session(
        stats,
        "candidates",
        options.batches,
        options.batch_size,
        runner.exit_on_first_match,
        |control| {
            super::device::run(&config, &control, |r, p, start, count| {
                engine.evaluate(r, p, &[], start, count)
            })
        },
    );
    engine.print_timings("Metal RSA");
    result
}

#[path = "../../../../kernels/rsa-modulus/src/contract.rs"]
mod contract;
pub type RsaTransport = Transport<contract::RsaModulus>;
