use crate::args::Command;
use crate::common::GlobalStats;
use crate::modes;
use crate::runner::Runner;
use std::error::Error;
use std::sync::Arc;

pub struct CpuRunner {
    num_threads: usize,
}

impl CpuRunner {
    pub fn new() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self { num_threads }
    }
}

impl Runner for CpuRunner {
    fn device_count(&self) -> usize {
        self.num_threads
    }

    fn run(&self, command: &Command, stats: Arc<GlobalStats>) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Starting CPU mode with {} threads", self.num_threads);

        match command {
            Command::SolanaVanity { prefix, suffix } => {
                modes::solana::cpu::run(self.num_threads, prefix.clone(), suffix.clone(), stats)
            }
            Command::BitcoinVanity { prefix, suffix } => {
                modes::bitcoin::cpu::run(self.num_threads, prefix.clone(), suffix.clone(), stats)
            }
            Command::EthereumVanity { prefix, suffix } => {
                modes::ethereum::cpu::run(self.num_threads, prefix.clone(), suffix.clone(), stats)
            }
            Command::Shallenge { username, target_hash } => {
                let target_hash_bytes = hex::decode(target_hash)?;
                modes::shallenge::cpu::run(self.num_threads, username.clone(), target_hash_bytes, stats)
            }
        }
    }
}
