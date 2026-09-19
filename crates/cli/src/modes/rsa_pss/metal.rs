use crate::runner::{
    RunResult,
    metal::{MetalRunner, transport::Transport},
    progress::GlobalStats,
    session::run_device_session,
};
use std::sync::Arc;

pub fn run(
    runner: &MetalRunner,
    args: &super::args::RsaPssArgs,
    stats: Arc<GlobalStats>,
) -> RunResult {
    let options = &runner.options;
    if options.seed.is_some() {
        return Err(
            "RSA-PSS uses OS cryptographic entropy for random salt; --seed is unsupported".into(),
        );
    }
    let config = args.config(1)?;
    let mut engine = RsaPssTransport::load(
        &runner.artifacts("rsa-pss"),
        options.batch_size,
        options.threads_per_group as usize,
        options.verify,
    )?;
    let unit = if matches!(config.source, super::PssSource::Salt { .. }) {
        "salts"
    } else {
        "messages"
    };
    let result = run_device_session(
        stats,
        unit,
        options.batches,
        options.batch_size,
        runner.exit_on_first_match,
        |control| {
            super::run_device(
                &config,
                control,
                &mut |request, pattern, message, start, count| {
                    engine.evaluate(request, pattern, message, start, count)
                },
            )
            .map(|_| ())
        },
    );
    engine.print_timings("Metal RSA-PSS");
    result
}

#[path = "../../../../kernels/rsa-pss/src/contract.rs"]
mod contract;
pub type RsaPssTransport = Transport<contract::RsaPss>;
