use crate::runner::{
    RunResult,
    metal::{MetalRunner, transport::Transport},
    progress::GlobalStats,
    session::run_device_session,
};
use std::sync::Arc;

pub fn run(
    runner: &MetalRunner,
    args: &super::args::P256SignatureArgs,
    stats: Arc<GlobalStats>,
) -> RunResult {
    let options = &runner.options;
    if options.seed.is_some() {
        return Err("p256-signature uses OS cryptographic entropy; --seed is not supported".into());
    }
    let config = args.config(1)?;
    config.validate()?;
    let mut engine = P256SignatureTransport::load(
        &runner.artifacts("p256-signature"),
        options.batch_size,
        options.threads_per_group as usize,
        options.verify,
    )?;
    let result = run_device_session(
        stats,
        if matches!(config.source, super::SearchSource::Message { .. }) {
            "messages"
        } else {
            "nonces"
        },
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
    eprintln!(
        "Metal: {} launches; load {:.3} ms; dispatch {:.3} ms; validation {:.3} ms",
        engine.launches,
        engine.load_time.as_secs_f64() * 1000.,
        engine.dispatch_time.as_secs_f64() * 1000.,
        engine.verification_time.as_secs_f64() * 1000.
    );
    result
}

#[path = "../../../../kernels/p256-signature/src/contract.rs"]
mod contract;
pub type P256SignatureTransport = Transport<contract::P256Signature>;
