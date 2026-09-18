//! Direct LLVM→AIR application backend.
use crate::{
    args::Command,
    runner::{RunResult, Runner, progress::GlobalStats},
};
use std::{path::PathBuf, sync::Arc};
pub mod transport;

#[derive(clap::Args, Clone)]
pub struct MetalOptions {
    /// Bundle from scripts/build-metal.py; defaults to target/metal/<mode>.
    #[arg(long, global = true)]
    pub metal_artifacts: Option<PathBuf>,
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
    pub(crate) fn artifacts(&self, mode: &str) -> PathBuf {
        self.options
            .metal_artifacts
            .clone()
            .unwrap_or_else(|| PathBuf::from("target/metal").join(mode))
    }
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
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity(args) => crate::modes::ethereum::metal::run(self, args, stats),
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity(args) => crate::modes::bitcoin::metal::run(self, args, stats),
            #[cfg(feature = "solana")]
            Command::SolanaVanity(args) => crate::modes::solana::metal::run(self, args, stats),
            #[allow(unreachable_patterns)]
            _ => Err("the Metal backend supports Shallenge, Ethereum, Bitcoin, and Solana".into()),
        }
    }
}

#[path = "../../../../kernels/shallenge/metal/src/contract.rs"]
mod contract;

#[cfg(feature = "ethereum")]
#[path = "../../../../kernels/ethereum/metal/src/contract.rs"]
mod ethereum_contract;

#[cfg(feature = "bitcoin")]
#[path = "../../../../kernels/bitcoin/metal/src/contract.rs"]
mod bitcoin_contract;

#[cfg(feature = "solana")]
#[path = "../../../../kernels/solana/metal/src/contract.rs"]
mod solana_contract;
