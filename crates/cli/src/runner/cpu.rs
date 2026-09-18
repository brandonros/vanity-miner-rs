use crate::args::Command;
#[allow(unused_imports)]
use crate::modes;
use crate::runner::Runner;
use crate::runner::progress::GlobalStats;
use std::error::Error;
use std::sync::Arc;

pub struct CpuRunner {
    pub(crate) exit_on_first_match: bool,
    num_threads: usize,
}

impl CpuRunner {
    pub fn new(threads: Option<usize>) -> Self {
        let num_threads = threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        });
        Self {
            num_threads,
            exit_on_first_match: false,
        }
    }
}

impl Runner for CpuRunner {
    fn set_exit_on_first_match(&mut self, enabled: bool) {
        self.exit_on_first_match = enabled;
    }
    fn device_count(&self) -> usize {
        self.num_threads
    }

    fn run(
        &self,
        command: &Command,
        stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Starting CPU mode with {} threads", self.num_threads);
        let _ = &stats;

        match *command {
            #[cfg(feature = "rsa-modulus")]
            Command::RsaModulusVanity(ref args) => modes::rsa_modulus::cpu::run(
                args,
                self.num_threads,
                stats,
                self.exit_on_first_match,
            ),
            #[cfg(feature = "rsa-pss")]
            Command::RsaPssSignatureVanity(ref args) => {
                modes::rsa_pss::cpu::run(args, self.num_threads, stats, self.exit_on_first_match)
            }
            #[cfg(feature = "p256-public-key")]
            Command::P256PublicKeyVanity(ref args) => modes::p256_public_key::cpu::run(
                args,
                self.num_threads,
                stats,
                self.exit_on_first_match,
            ),
            #[cfg(feature = "p256-signature")]
            Command::P256SignatureVanity(ref args) => modes::p256_signature::cpu::run(
                args,
                self.num_threads,
                stats,
                self.exit_on_first_match,
            ),
            #[cfg(feature = "solana")]
            Command::SolanaVanity(ref args) => {
                modes::solana::cpu::run(args, self.num_threads, stats, self.exit_on_first_match)
            }
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity(ref args) => {
                modes::bitcoin::cpu::run(args, self.num_threads, stats, self.exit_on_first_match)
            }
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity(ref args) => {
                modes::ethereum::cpu::run(args, self.num_threads, stats, self.exit_on_first_match)
            }
            #[cfg(feature = "shallenge")]
            Command::Shallenge(ref args) => {
                modes::shallenge::cpu::run(args, self.num_threads, stats, self.exit_on_first_match)
            }
            #[cfg(feature = "self_test_support")]
            Command::SelfTest(ref args) => modes::self_test::cpu::run(args),
        }
    }
}
