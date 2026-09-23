use crate::args::Command;
#[cfg(any(
    feature = "solana",
    feature = "bitcoin",
    feature = "ethereum",
    feature = "shallenge",
    feature = "self_test_support",
    feature = "crypto-cli"
))]
use crate::modes;
#[cfg(feature = "bitcoin")]
use crate::modes::bitcoin::args::BitcoinArgs;
#[cfg(feature = "ethereum")]
use crate::modes::ethereum::args::EthereumArgs;
#[cfg(feature = "shallenge")]
use crate::modes::shallenge::args::ShallengeArgs;
#[cfg(feature = "solana")]
use crate::modes::solana::args::SolanaArgs;
use crate::runner::Runner;
use crate::runner::progress::GlobalStats;
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

    fn run(
        &self,
        command: &Command,
        stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!(
            "Starting CPU mode with {} threads",
            command.details().cpu_threads.unwrap_or(self.num_threads)
        );
        let _ = &stats;

        match command {
            #[cfg(feature = "rsa-pss")]
            Command::RsaPssSignatureVanity(args) => {
                modes::rsa_pss::cpu::run(args, self.num_threads, stats)
            }
            #[cfg(feature = "p256-public-key")]
            Command::P256PublicKeyVanity(args) => {
                modes::p256_public_key::cpu::run(args, self.num_threads, stats)
            }
            #[cfg(feature = "p256-signature")]
            Command::P256SignatureVanity(args) => {
                modes::p256_signature::cpu::run(args, self.num_threads, stats)
            }
            #[cfg(feature = "solana")]
            Command::SolanaVanity(SolanaArgs { prefix, suffix }) => {
                modes::solana::cpu::run(self.num_threads, prefix.clone(), suffix.clone(), stats)
            }
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity(BitcoinArgs { prefix, suffix }) => {
                modes::bitcoin::cpu::run(self.num_threads, prefix.clone(), suffix.clone(), stats)
            }
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity(EthereumArgs { prefix, suffix }) => {
                modes::ethereum::cpu::run(self.num_threads, prefix.clone(), suffix.clone(), stats)
            }
            #[cfg(feature = "shallenge")]
            Command::Shallenge(ShallengeArgs {
                username,
                target_hash,
            }) => {
                let target_hash_bytes = hex::decode(target_hash)?;
                modes::shallenge::cpu::run(
                    self.num_threads,
                    username.clone(),
                    target_hash_bytes,
                    stats,
                )
            }
            #[cfg(feature = "self_test_support")]
            Command::SelfTest(args) => modes::self_test::cpu::run(args),
        }
    }
}
