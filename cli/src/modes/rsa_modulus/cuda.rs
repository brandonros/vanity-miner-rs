//! One mining launch resumes exclusively owned per-thread tasks on each device.
use crate::runner::cuda::buffers::Records;
use crate::runner::cuda::context::GpuContext;
use cust::{function::Function, launch};
use logic::{
    modes::rsa_modulus::{Counts, Pair, SearchConfig, Task},
    search::hex_pattern::HexPattern,
};
use zeroize::Zeroizing;

pub(crate) struct RsaPipeline<'a> {
    gpu: &'a GpuContext,
    mine: Function<'a>,
    config: Records<'a, SearchConfig>,
    pattern: Records<'a, HexPattern>,
    tasks: Records<'a, Task>,
    pairs: Records<'a, Pair>,
    counts: Records<'a, Counts>,
    steps: u32,
    capacity: u32,
}

impl<'a> RsaPipeline<'a> {
    pub fn new(
        gpu: &'a GpuContext,
        config: &SearchConfig,
        pattern: &HexPattern,
        capacity: u32,
        steps: u32,
    ) -> Result<Self, String> {
        logic::modes::rsa_modulus::launch_work(capacity, steps)?;
        let mine = gpu
            .module()?
            .get_function(logic::modes::rsa_modulus::ENTRY)
            .map_err(|e| format!("RSA miner requires rebuilt single-entry PTX: {e}"))?;
        Ok(Self {
            gpu,
            mine,
            steps,
            config: Records::from_slice(std::slice::from_ref(config), &gpu.stream)?,
            pattern: Records::from_slice(std::slice::from_ref(pattern), &gpu.stream)?,
            tasks: Records::zeroed(capacity as usize, &gpu.stream)?,
            pairs: Records::zeroed(capacity as usize, &gpu.stream)?,
            counts: Records::zeroed(1, &gpu.stream)?,
            capacity,
        })
    }

    pub fn cycle(&mut self, start: u64) -> Result<(Counts, Zeroizing<Vec<Pair>>), String> {
        let capacity = self.capacity;
        let steps = self.steps;
        let work = logic::modes::rsa_modulus::launch_work(capacity, steps)?;
        start
            .checked_add(u64::from(work) - 1)
            .ok_or("RSA task counter exhausted")?;
        self.counts.clear_async()?;
        let stream = &self.gpu.stream;
        let threads = self.gpu.threads_per_block as u32;
        let blocks = capacity.div_ceil(threads);
        let mine = &self.mine;
        unsafe {
            launch!(mine<<<blocks, threads, 0, stream>>>(self.config.pointer(), self.pattern.pointer(), self.tasks.pointer(), self.pairs.pointer(), start, capacity, steps, self.counts.pointer())).map_err(|e| e.to_string())?;
        }
        stream.synchronize().map_err(|e| e.to_string())?;
        let counts = self.counts.read(1)?[0];
        counts.validate(capacity, steps)?;
        let pairs = self.pairs.read(counts.matches as usize)?;
        self.pairs.clear_prefix(counts.matches as usize)?;
        Ok((counts, pairs))
    }
}

use crate::runner::RunResult;
pub fn run(
    args: &crate::modes::rsa_modulus::args::RsaModulusArgs,
    gpu: &crate::runner::cuda::context::GpuContext,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
    control: std::sync::Arc<crate::runner::session::SearchControl>,
    stages: &super::pipeline::StageStats,
) -> RunResult {
    let mut engine = None;
    let config = args.config(1)?;
    stats.set_unit("factor candidates (p + q)");
    crate::modes::rsa_modulus::pipeline::run(
        &config,
        &control,
        stages,
        args.steps_per_launch,
        |r, p, start, capacity| {
            if engine.is_none() {
                engine = Some(RsaPipeline::new(
                    gpu,
                    r,
                    p,
                    capacity,
                    args.steps_per_launch,
                )?);
            }
            engine.as_mut().unwrap().cycle(start)
        },
    )
    .map_err(Into::into)
}
