//! CuMetal host backend: consume prebuilt PTX without a local NVIDIA toolchain.
#[cfg(feature = "crypto-cli")]
pub(crate) mod batch_transport;
pub(crate) mod driver;
use crate::{args::Command, common::GlobalStats, runner::Runner};
use driver::Driver;
use std::{rc::Rc, sync::Arc};
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
        #[cfg(feature = "self_test_support")]
        if matches!(command, Command::SelfTest) {
            return self.self_tests(&driver);
        }
        #[cfg(feature = "crypto-cli")]
        if let Some(result) = self.crypto_search(command, &driver, stats.clone()) {
            return result;
        }
        #[cfg(any(
            feature = "solana",
            feature = "bitcoin",
            feature = "ethereum",
            feature = "shallenge"
        ))]
        {
            return self.address_search(command, &driver, stats);
        }
        #[allow(unreachable_code)]
        Err("no search mode enabled".into())
    }
}

#[cfg(feature = "crypto-cli")]
impl CumetalRunner {
    pub(super) fn crypto_search(
        &self,
        command: &Command,
        driver: &Rc<Driver>,
        stats: Arc<GlobalStats>,
    ) -> Option<Result<(), Error>> {
        match command {
            #[cfg(feature = "rsa-modulus")]
            Command::RsaModulusVanity(args) => Some(crate::modes::rsa_modulus::cumetal::run(
                self, args, driver, stats,
            )),
            #[cfg(feature = "p256-public-key")]
            Command::P256PublicKeyVanity(args) => Some(
                crate::modes::p256_public_key::cumetal::run(self, args, driver, stats),
            ),
            #[cfg(feature = "p256-signature")]
            Command::P256SignatureVanity(args) => Some(crate::modes::p256_signature::cumetal::run(
                self, args, driver, stats,
            )),
            #[cfg(feature = "rsa-pss")]
            Command::RsaPssSignatureVanity(args) => Some(crate::modes::rsa_pss::cumetal::run(
                self, args, driver, stats,
            )),
            #[allow(unreachable_patterns)]
            _ => None,
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
#[cfg(feature = "self_test_support")]
mod self_test;
