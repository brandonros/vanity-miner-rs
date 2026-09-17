use super::{args::ShallengeArgs, shared_best_hash::SharedBestHash};
use crate::runner::{
    RunResult,
    metal::{MetalRunner, transport::ShallengeTransport},
    progress::GlobalStats,
    session::run_device_session,
};
use std::sync::{Arc, RwLock};

pub fn run(runner: &MetalRunner, args: &ShallengeArgs, stats: Arc<GlobalStats>) -> RunResult {
    let best = Arc::new(RwLock::new(SharedBestHash::new(
        hex::decode(&args.target_hash)?
            .try_into()
            .map_err(|_| "invalid target width")?,
    )));
    let options = &runner.options;
    let mut engine = ShallengeTransport::load(
        &options.metal_artifacts,
        options.batch_size,
        options.threads_per_group as usize,
        options.verify,
    )?;
    let result = run_device_session(
        stats,
        "nonces",
        options.batches,
        options.batch_size,
        |control| {
            super::device::search(
                &args.username,
                best,
                options.seed,
                &control,
                |r, p, m, start, count| engine.evaluate(r, p, m, start, count),
            )
            .map(|_| ())
        },
    );
    eprintln!(
        "Metal: {} launches; dispatch {:.3} ms; validation {:.3} ms",
        engine.launches,
        engine.dispatch_time.as_secs_f64() * 1000.,
        engine.verification_time.as_secs_f64() * 1000.
    );
    result
}
