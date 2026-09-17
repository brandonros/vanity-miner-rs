use crate::runner::{
    RunResult,
    cuda::{batch::CandidateBatch, context::GpuContext},
    progress::GlobalStats,
    session::SearchControl,
};
use std::sync::Arc;

pub fn run(
    ordinal: usize,
    prefix: String,
    suffix: String,
    gpu: &GpuContext,
    stats: Arc<GlobalStats>,
    control: Arc<SearchControl>,
) -> RunResult {
    gpu.print_launch_info(ordinal, "solana", control.batch_size());
    stats.set_unit("keys");
    let mut engine = None;
    super::device::search(&prefix, &suffix, None, &control, |r, p, m, start, count| {
        if engine.is_none() {
            engine = Some(CandidateBatch::new(gpu, super::device::ENTRY, r, p, m)?);
        }
        engine.as_mut().unwrap().evaluate(start, count)
    })
    .map(|_| ())
    .map_err(Into::into)
}
