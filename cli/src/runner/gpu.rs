#[cfg(feature = "shallenge")]
use crate::common::SharedBestHash;
use crate::{
    args::Command,
    common::{GlobalStats, GpuContext},
    modes,
    runner::Runner,
};
#[cfg(feature = "shallenge")]
use std::sync::RwLock;
use std::{error::Error, sync::Arc};
use vanity_miner::search_control::SearchControl;

type RunResult = Result<(), Box<dyn Error + Send + Sync>>;
pub struct GpuRunner {
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
        Ok(Self { num_devices })
    }
    fn run_devices(
        &self,
        command: &Command,
        stats: Arc<GlobalStats>,
        control: Arc<SearchControl>,
    ) -> RunResult {
        #[cfg(feature = "shallenge")]
        let shared_best_hash = match command {
            Command::Shallenge { target_hash, .. } => {
                let bytes: [u8; 32] = hex::decode(target_hash)?
                    .try_into()
                    .map_err(|_| "invalid target hash width")?;
                Some(Arc::new(RwLock::new(SharedBestHash::new(bytes))))
            }
            #[allow(unreachable_patterns)]
            _ => None,
        };
        vanity_miner::device_workers::run(
            self.num_devices,
            control.clone(),
            stats.clone(),
            is_crypto_search(command),
            |ordinal| match command {
                #[cfg(feature = "self_test_support")]
                Command::SelfTest => GpuContext::for_self_test(ordinal),
                #[allow(unreachable_patterns)]
                _ => GpuContext::new(ordinal, command.ptx_module()),
            },
            |gpu, ordinal| match command {
                #[cfg(feature = "p256-public-key")]
                Command::P256PublicKeyVanity(args) => {
                    modes::p256_public::gpu::run(args, gpu, stats.clone(), control.clone())
                }
                #[cfg(feature = "p256-signature")]
                Command::P256SignatureVanity(args) => {
                    modes::p256_signature::gpu::run(args, gpu, stats.clone(), control.clone())
                }
                #[cfg(feature = "rsa-pss")]
                Command::RsaPssSignatureVanity(args) => {
                    modes::rsa_pss_search::gpu::run(args, gpu, stats.clone(), control.clone())
                }
                #[cfg(feature = "rsa-modulus")]
                Command::RsaModulusVanity(args) => {
                    modes::rsa_modulus::gpu::run(args, gpu, stats.clone(), control.clone())
                }
                #[cfg(feature = "solana")]
                Command::SolanaVanity { prefix, suffix } => modes::solana::gpu::run(
                    ordinal,
                    prefix.clone(),
                    suffix.clone(),
                    gpu,
                    stats.clone(),
                    control.clone(),
                ),
                #[cfg(feature = "bitcoin")]
                Command::BitcoinVanity { prefix, suffix } => modes::bitcoin::gpu::run(
                    ordinal,
                    prefix.clone(),
                    suffix.clone(),
                    gpu,
                    stats.clone(),
                    control.clone(),
                ),
                #[cfg(feature = "ethereum")]
                Command::EthereumVanity { prefix, suffix } => modes::ethereum::gpu::run(
                    ordinal,
                    prefix.clone(),
                    suffix.clone(),
                    gpu,
                    stats.clone(),
                    control.clone(),
                ),
                #[cfg(feature = "shallenge")]
                Command::Shallenge { username, .. } => modes::shallenge::gpu::run(
                    ordinal,
                    username.clone(),
                    shared_best_hash.clone().unwrap(),
                    gpu,
                    stats.clone(),
                    control.clone(),
                ),
                #[cfg(feature = "self_test_support")]
                Command::SelfTest => modes::self_test::gpu::run(ordinal, &gpu),
            },
        )
    }
}
impl Runner for GpuRunner {
    fn device_count(&self) -> usize {
        self.num_devices
    }
    fn run(&self, command: &Command, stats: Arc<GlobalStats>) -> RunResult {
        let control = Arc::new(SearchControl::with_stats(stats.clone()));
        #[cfg(feature = "crypto-cli")]
        if is_crypto_search(command) {
            let threads = GpuContext::configured_threads_per_block()? as u32;
            let batch_size = if let Ok(value) = std::env::var("CRYPTO_BATCH_SIZE") {
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
                "CUDA crypto launch: up to {batch_size} candidates, {threads} threads/block, {} devices",
                self.num_devices
            );
        }
        self.run_devices(command, stats, control)
    }
}

fn is_crypto_search(command: &Command) -> bool {
    match command {
        #[cfg(feature = "p256-public-key")]
        Command::P256PublicKeyVanity(..) => true,
        #[cfg(feature = "p256-signature")]
        Command::P256SignatureVanity(..) => true,
        #[cfg(feature = "rsa-pss")]
        Command::RsaPssSignatureVanity(..) => true,
        #[cfg(feature = "rsa-modulus")]
        Command::RsaModulusVanity(..) => true,
        #[allow(unreachable_patterns)]
        _ => false,
    }
}
