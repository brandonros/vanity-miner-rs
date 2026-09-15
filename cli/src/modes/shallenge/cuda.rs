use super::shared_best_hash::SharedBestHash;
use crate::runner::{
    RunResult,
    cuda::{batch::CandidateBatch, context::GpuContext},
    progress::GlobalStats,
    session::SearchControl,
};
use std::sync::{Arc, RwLock};

pub fn run(
    ordinal: usize,
    username: String,
    best: Arc<RwLock<SharedBestHash>>,
    gpu: &GpuContext,
    stats: Arc<GlobalStats>,
    control: Arc<SearchControl>,
) -> RunResult {
    gpu.print_launch_info(ordinal, "shallenge", control.batch_size());
    stats.set_unit("nonces");
    let mut engine = None;
    super::device::search(&username, best, None, &control, |r, p, m, start, count| {
        if engine.is_none() {
            engine = Some(CandidateBatch::new(gpu, super::device::ENTRY, r, p, m)?);
        }
        let engine = engine.as_mut().unwrap();
        engine.update_pattern(p)?;
        engine.evaluate(start, count)
    })
    .map(|_| ())
    .map_err(Into::into)
}
