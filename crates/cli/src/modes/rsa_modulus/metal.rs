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
    if runner.options.seed.is_some() {
        return Err("RSA uses OS cryptographic entropy; --seed is not supported".into());
    }
    let config = args.config(1)?;
    let stages = super::pipeline::StageStats::attach(&stats)?;
    let mut engine = None;
    let result = run_device_session(
        stats,
        "factor candidates (p + q)",
        runner.options.batches,
        runner.options.batch_size,
        |control| {
            super::pipeline::run(
                &config,
                &control,
                &stages,
                args.steps_per_launch,
                |r, p, start, capacity| {
                    if engine.is_none() {
                        engine = Some(RsaTransport::load(
                            &runner.artifacts("rsa-modulus"),
                            r,
                            p,
                            capacity,
                            args.steps_per_launch,
                            runner.options.threads_per_group as usize,
                            runner.options.verify,
                        )?);
                    }
                    engine.as_mut().unwrap().cycle(r, p, start)
                },
            )
        },
    );
    if let Some(engine) = engine {
        eprintln!(
            "Metal RSA: {} launches; load {:.3} ms; dispatch {:.3} ms; validation {:.3} ms",
            engine.launches,
            engine.load_time.as_secs_f64() * 1000.,
            engine.dispatch_time.as_secs_f64() * 1000.,
            engine.verification_time.as_secs_f64() * 1000.
        );
    }
    result
}
