//! Four ordered stages share one persistent workspace per CUDA device.
use crate::runner::cuda::buffers::Records;
use crate::runner::cuda::context::GpuContext;
use cust::{function::Function, launch, memory::DeviceBuffer};
use logic::{
    modes::rsa_modulus::pipeline::{Counts, Pair, SearchConfig, Task},
    search::hex_pattern::HexPattern,
};
use zeroize::Zeroizing;

pub(crate) struct RsaPipeline<'a> {
    gpu: &'a GpuContext,
    generate: Function<'a>,
    ranges: Function<'a>,
    search: Function<'a>,
    advance: Function<'a>,
    config: Records<'a, SearchConfig>,
    pattern: Records<'a, HexPattern>,
    tasks: Records<'a, Task>,
    pairs: Records<'a, Pair>,
    counts: Records<'a, Counts>,
    active: DeviceBuffer<u32>,
    winners: DeviceBuffer<u32>,
    capacity: u32,
}

impl<'a> RsaPipeline<'a> {
    pub fn new(
        gpu: &'a GpuContext,
        config: &SearchConfig,
        pattern: &HexPattern,
        capacity: u32,
    ) -> Result<Self, String> {
        if !(1..=1_048_576).contains(&capacity) {
            return Err("invalid RSA workspace capacity".into());
        }
        let module = gpu.module()?;
        let function = |name| {
            module
                .get_function(name)
                .map_err(|e| format!("RSA pipeline requires rebuilt PTX ({name}): {e}"))
        };
        Ok(Self {
            gpu,
            generate: function("kernel_rsa_generate_v3")?,
            ranges: function("kernel_rsa_ranges_v3")?,
            search: function("kernel_rsa_search_v3")?,
            advance: function("kernel_rsa_advance_v3")?,
            config: Records::from_slice(std::slice::from_ref(config), &gpu.stream)?,
            pattern: Records::from_slice(std::slice::from_ref(pattern), &gpu.stream)?,
            tasks: Records::zeroed(capacity as usize, &gpu.stream)?,
            pairs: Records::zeroed(capacity as usize, &gpu.stream)?,
            counts: Records::zeroed(1, &gpu.stream)?,
            active: DeviceBuffer::zeroed(capacity as usize).map_err(|e| e.to_string())?,
            winners: DeviceBuffer::zeroed(capacity as usize).map_err(|e| e.to_string())?,
            capacity,
        })
    }

    pub fn cycle(&mut self, start: u64) -> Result<(Counts, Zeroizing<Vec<Pair>>), String> {
        let capacity = self.capacity;
        start
            .checked_add(u64::from(capacity) - 1)
            .ok_or("RSA task counter exhausted")?;
        self.counts.clear_async()?;
        let stream = &self.gpu.stream;
        let threads = self.gpu.threads_per_block as u32;
        let blocks = capacity.div_ceil(threads);
        let (generate, ranges, search, advance) =
            (&self.generate, &self.ranges, &self.search, &self.advance);
        unsafe {
            launch!(generate<<<blocks, threads, 0, stream>>>(self.config.pointer(), self.tasks.pointer(), start, capacity, self.counts.pointer())).map_err(|e| e.to_string())?;
            launch!(ranges<<<blocks, threads, 0, stream>>>(self.config.pointer(), self.tasks.pointer(), self.active.as_device_ptr(), self.winners.as_device_ptr(), capacity, self.counts.pointer())).map_err(|e| e.to_string())?;
            launch!(search<<<blocks, threads, 0, stream>>>(self.config.pointer(), self.pattern.pointer(), self.tasks.pointer(), self.active.as_device_ptr(), self.winners.as_device_ptr(), self.pairs.pointer(), capacity, self.counts.pointer())).map_err(|e| e.to_string())?;
            launch!(advance<<<blocks, threads, 0, stream>>>(self.tasks.pointer(), self.active.as_device_ptr(), self.winners.as_device_ptr(), capacity, self.counts.pointer())).map_err(|e| e.to_string())?;
        }
        stream.synchronize().map_err(|e| e.to_string())?;
        let counts = self.counts.read(1)?[0];
        if counts.errors != 0 {
            return Err("RSA device pipeline reported an arithmetic or queue error".into());
        }
        if counts.p_tested > capacity
            || counts.p_accepted > counts.p_tested
            || counts.ranges > counts.p_accepted
            || counts.active > capacity
            || counts.q_tested > capacity
            || counts.matches > counts.active
            || counts.matches > counts.q_tested
        {
            return Err("RSA device pipeline returned invalid counters".into());
        }
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
    crate::modes::rsa_modulus::pipeline::run(&config, &control, stages, |r, p, start, capacity| {
        if engine.is_none() {
            engine = Some(RsaPipeline::new(gpu, r, p, capacity)?);
        }
        engine.as_mut().unwrap().cycle(start)
    })
    .map_err(Into::into)
}
