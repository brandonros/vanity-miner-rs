//! Direct LLVM→AIR backend. Its first application workload is Shallenge.
use crate::{
    args::Command,
    runner::{RunResult, Runner, progress::GlobalStats},
};
use std::{path::PathBuf, sync::Arc};
pub mod transport;

#[derive(clap::Args, Clone)]
pub struct MetalOptions {
    /// Kernel bundle produced by scripts/build-metal-shallenge.py.
    #[arg(long, global = true, default_value = "target/metal/shallenge")]
    pub metal_artifacts: PathBuf,
    /// Stop after this many launches; omitted means continuous search.
    #[arg(long, global = true, value_parser = clap::value_parser!(u64).range(1..))]
    pub batches: Option<u64>,
    #[arg(long, global = true, default_value_t = 4096, value_parser = clap::value_parser!(u32).range(1..=1048576))]
    pub batch_size: u32,
    #[arg(long, global = true, default_value_t = 64, value_parser = clap::value_parser!(u32).range(1..=1024))]
    pub threads_per_group: u32,
    /// Deterministic starting seed, advanced according to the global counter.
    #[arg(long, global = true)]
    pub seed: Option<u64>,
    /// Return every lane's result and compare it with the CPU, including misses.
    #[arg(long, global = true)]
    pub verify: bool,
}

pub struct MetalRunner {
    pub(crate) options: MetalOptions,
}
impl MetalRunner {
    pub fn new(options: MetalOptions) -> Result<Self, String> {
        if options.batch_size == 0
            || options.batch_size > 1_048_576
            || options.threads_per_group == 0
        {
            return Err("invalid Metal dispatch size".into());
        }
        Ok(Self { options })
    }
}
impl Runner for MetalRunner {
    fn device_count(&self) -> usize {
        1
    }
    fn run(&self, command: &Command, stats: Arc<GlobalStats>) -> RunResult {
        match command {
            Command::Shallenge(args) => crate::modes::shallenge::metal::run(self, args, stats),
            #[allow(unreachable_patterns)]
            _ => Err("the Metal backend currently supports only Shallenge".into()),
        }
    }
}

#[path = "../../../../kernels/shallenge/metal/src/contract.rs"]
mod contract;
