//! CuMetal host backend: consume prebuilt PTX without a local NVIDIA toolchain.
#[cfg(feature = "bitcoin")]
use crate::modes::bitcoin::args::BitcoinArgs;
#[cfg(feature = "ethereum")]
use crate::modes::ethereum::args::EthereumArgs;
#[cfg(feature = "shallenge")]
use crate::modes::shallenge::args::ShallengeArgs;
#[cfg(feature = "solana")]
use crate::modes::solana::args::SolanaArgs;
#[cfg(feature = "crypto-cli")]
pub(crate) mod batch_transport;
pub(crate) mod driver;
use crate::{args::Command, runner::Runner, runner::progress::GlobalStats};
use driver::Driver;
use std::sync::Arc;
pub type Error = Box<dyn std::error::Error + Send + Sync>;

pub struct CumetalRunner {
    pub(crate) options: CumetalOptions,
}
impl CumetalRunner {
    pub fn new(options: CumetalOptions) -> Result<Self, Error> {
        if options.ptx.is_none() && options.module_dir.is_none() {
            return Err("Specify --ptx or --module-dir for CuMetal".into());
        }
        options
            .blocks
            .checked_mul(options.threads_per_block)
            .ok_or("launch size overflow")?;
        Ok(Self { options })
    }
}

impl Runner for CumetalRunner {
    fn device_count(&self) -> usize {
        1
    }
    fn run(&self, command: &Command, stats: Arc<GlobalStats>) -> Result<(), Error> {
        let driver = Driver::open(&self.options.cumetal_library)?;
        match command {
            #[cfg(feature = "rsa-modulus")]
            Command::RsaModulusVanity(args) => {
                crate::modes::rsa_modulus::cumetal::run(self, args, &driver, stats)
            }
            #[cfg(feature = "rsa-pss")]
            Command::RsaPssSignatureVanity(args) => {
                crate::modes::rsa_pss::cumetal::run(self, args, &driver, stats)
            }
            #[cfg(feature = "p256-public-key")]
            Command::P256PublicKeyVanity(args) => {
                crate::modes::p256_public_key::cumetal::run(self, args, &driver, stats)
            }
            #[cfg(feature = "p256-signature")]
            Command::P256SignatureVanity(args) => {
                crate::modes::p256_signature::cumetal::run(self, args, &driver, stats)
            }
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity(BitcoinArgs { prefix, suffix }) => {
                crate::modes::bitcoin::cumetal::run(self, prefix, suffix, &driver, stats)
            }
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity(EthereumArgs { prefix, suffix }) => {
                crate::modes::ethereum::cumetal::run(self, prefix, suffix, &driver, stats)
            }
            #[cfg(feature = "solana")]
            Command::SolanaVanity(SolanaArgs { prefix, suffix }) => {
                crate::modes::solana::cumetal::run(self, prefix, suffix, &driver, stats)
            }
            #[cfg(feature = "shallenge")]
            Command::Shallenge(ShallengeArgs {
                username,
                target_hash,
            }) => {
                crate::modes::shallenge::cumetal::run(self, username, target_hash, &driver, stats)
            }
            #[cfg(feature = "self_test_support")]
            Command::SelfTest => crate::modes::self_test::cumetal::run(self, &driver),
        }
    }
}

mod options;
pub use options::CumetalOptions;
#[cfg(any(
    feature = "solana",
    feature = "bitcoin",
    feature = "ethereum",
    feature = "shallenge"
))]
pub(crate) mod address_transport;
mod module;
