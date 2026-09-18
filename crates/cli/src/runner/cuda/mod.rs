//! CUDA resource management and command dispatch.
#[cfg(feature = "bitcoin")]
use crate::modes::bitcoin::args::BitcoinArgs;
#[cfg(feature = "ethereum")]
use crate::modes::ethereum::args::EthereumArgs;
#[cfg(feature = "shallenge")]
use crate::modes::shallenge::args::ShallengeArgs;
#[cfg(feature = "shallenge")]
use crate::modes::shallenge::shared_best_hash::SharedBestHash;
#[cfg(feature = "solana")]
use crate::modes::solana::args::SolanaArgs;
use crate::runner::session::SearchControl;
use crate::{
    args::Command,
    modes,
    runner::Runner,
    runner::{cuda::context::GpuContext, progress::GlobalStats},
};
#[cfg(feature = "shallenge")]
use std::sync::RwLock;
use std::{error::Error, sync::Arc};

type RunResult = Result<(), Box<dyn Error + Send + Sync>>;
pub struct GpuRunner {
    pub(crate) exit_on_first_match: bool,
    num_devices: usize,
}
impl GpuRunner {
    pub fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        cust::init(cust::CudaFlags::empty())?;
        let num_devices = cust::device::Device::num_devices()? as usize;
        if num_devices == 0 {
            return Err("no CUDA devices available".into());
        }
        println!("Found {num_devices} CUDA devices");
        Ok(Self {
            num_devices,
            exit_on_first_match: false,
        })
    }
    fn run_devices(
        &self,
        command: &Command,
        stats: Arc<GlobalStats>,
        control: Arc<SearchControl>,
    ) -> RunResult {
        #[cfg(feature = "shallenge")]
        let shared_best_hash = match command {
            Command::Shallenge(ShallengeArgs { target_hash, .. }) => {
                let bytes: [u8; 32] = hex::decode(target_hash)?
                    .try_into()
                    .map_err(|_| "invalid target hash width")?;
                Some(Arc::new(RwLock::new(SharedBestHash::new(bytes))))
            }
            #[allow(unreachable_patterns)]
            _ => None,
        };
        crate::runner::workers::device::run(
            self.num_devices,
            control.clone(),
            |ordinal| match command {
                #[cfg(feature = "self_test_support")]
                Command::SelfTest(_) => GpuContext::without_module(ordinal),
                #[allow(unreachable_patterns)]
                _ => GpuContext::new(
                    ordinal,
                    command
                        .details()
                        .cuda_module
                        .ok_or("self-tests select their own modules")?,
                ),
            },
            |gpu, ordinal| match command {
                #[cfg(feature = "p256-public-key")]
                Command::P256PublicKeyVanity(args) => {
                    modes::p256_public_key::cuda::run(args, gpu, stats.clone(), control.clone())
                }
                #[cfg(feature = "p256-signature")]
                Command::P256SignatureVanity(args) => {
                    modes::p256_signature::cuda::run(args, gpu, stats.clone(), control.clone())
                }
                #[cfg(feature = "rsa-pss")]
                Command::RsaPssSignatureVanity(args) => {
                    modes::rsa_pss::cuda::run(args, gpu, stats.clone(), control.clone())
                }
                #[cfg(feature = "rsa-modulus")]
                Command::RsaModulusVanity(args) => {
                    modes::rsa_modulus::cuda::run(args, gpu, stats.clone(), control.clone())
                }
                #[cfg(feature = "solana")]
                Command::SolanaVanity(SolanaArgs { prefix, suffix }) => modes::solana::cuda::run(
                    ordinal,
                    prefix.clone(),
                    suffix.clone(),
                    gpu,
                    stats.clone(),
                    control.clone(),
                ),
                #[cfg(feature = "bitcoin")]
                Command::BitcoinVanity(BitcoinArgs { prefix, suffix }) => {
                    modes::bitcoin::cuda::run(
                        ordinal,
                        prefix.clone(),
                        suffix.clone(),
                        gpu,
                        stats.clone(),
                        control.clone(),
                    )
                }
                #[cfg(feature = "ethereum")]
                Command::EthereumVanity(EthereumArgs { prefix, suffix }) => {
                    modes::ethereum::cuda::run(
                        ordinal,
                        prefix.clone(),
                        suffix.clone(),
                        gpu,
                        stats.clone(),
                        control.clone(),
                    )
                }
                #[cfg(feature = "shallenge")]
                Command::Shallenge(ShallengeArgs { username, .. }) => modes::shallenge::cuda::run(
                    ordinal,
                    username.clone(),
                    shared_best_hash.clone().unwrap(),
                    gpu,
                    stats.clone(),
                    control.clone(),
                ),
                #[cfg(feature = "self_test_support")]
                Command::SelfTest(args) => modes::self_test::cuda::run(ordinal, &gpu, args),
            },
        )
    }
}
impl Runner for GpuRunner {
    fn set_exit_on_first_match(&mut self, enabled: bool) {
        self.exit_on_first_match = enabled;
    }
    fn device_count(&self) -> usize {
        self.num_devices
    }
    fn run(&self, command: &Command, stats: Arc<GlobalStats>) -> RunResult {
        let control = Arc::new(SearchControl::with_stats(stats.clone()));
        if self.exit_on_first_match {
            control.set_exit_on_first_match();
        }
        if command.details().cuda_module.is_some() {
            control.set_continuous();
            let threads = GpuContext::configured_threads_per_block()? as u32;
            let batch_size = if let Ok(value) = std::env::var("BATCH_SIZE") {
                value.parse::<u32>()?
            } else {
                // Four blocks per SM on the largest selected GPU, using the
                // same configured block size as the other search modes.
                let mut sms = 1u32;
                for ordinal in 0..self.num_devices {
                    let device = cust::device::Device::get_device(ordinal as u32)?;
                    sms = sms.max(
                        device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)?
                            as u32,
                    );
                }
                sms.saturating_mul(4).saturating_mul(threads).min(1_048_576)
            };
            control.set_batch_size(batch_size)?;
            let cancellation = control.clone();
            ctrlc::set_handler(move || cancellation.interrupt())
                .map_err(|_| "could not install Ctrl-C handler")?;
            println!(
                "CUDA launch: up to {batch_size} candidates, {threads} threads/block, {} devices",
                self.num_devices
            );
        }
        self.run_devices(command, stats, control)
    }
}

#[cfg(any(
    feature = "solana",
    feature = "bitcoin",
    feature = "ethereum",
    feature = "shallenge",
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-pss",
    feature = "rsa-modulus"
))]
pub(crate) mod batch;
pub(crate) mod buffers;
pub(crate) mod context;
pub(crate) mod module;
