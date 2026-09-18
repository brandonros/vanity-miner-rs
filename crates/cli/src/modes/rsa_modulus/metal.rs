use crate::runner::{
    RunResult,
    metal::{MetalRunner, rsa::RsaTransport},
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
        |control| {
            super::pipeline::run(&config, &control, |r, p, start, count| {
                engine.evaluate(r, p, &[], start, count)
            })
        },
    );
    eprintln!(
        "Metal RSA: {} launches; load {:.3} ms; dispatch/transfer {:.3} ms; validation {:.3} ms",
        engine.launches,
        engine.load_time.as_secs_f64() * 1000.,
        engine.dispatch_time.as_secs_f64() * 1000.,
        engine.verification_time.as_secs_f64() * 1000.
    );
    result
}
