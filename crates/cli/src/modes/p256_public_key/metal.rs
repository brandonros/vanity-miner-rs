use crate::runner::{
    RunResult,
    metal::{MetalRunner, p256::P256PublicTransport},
    progress::GlobalStats,
    session::run_device_session,
};
use std::sync::Arc;

pub fn run(
    runner: &MetalRunner,
    args: &super::args::P256PublicArgs,
    stats: Arc<GlobalStats>,
) -> RunResult {
    let options = &runner.options;
    if options.seed.is_some() {
        return Err(
            "p256-public-key uses OS cryptographic entropy; --seed is not supported".into(),
        );
    }
    let config = args.config(1);
    config.validate()?;
    let mut engine = P256PublicTransport::load(
        &runner.artifacts("p256-public-key"),
        options.batch_size,
        options.threads_per_group as usize,
        options.verify,
    )?;
    let result = run_device_session(
        stats,
        "keys",
        options.batches,
        options.batch_size,
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
