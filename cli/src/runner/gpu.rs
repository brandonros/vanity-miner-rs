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
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for ordinal in 0..self.num_devices {
                let stats = stats.clone();
                let control = control.clone();
                #[cfg(feature = "shallenge")]
                let shared_best_hash = shared_best_hash.clone();
                handles.push(scope.spawn(move || -> RunResult {
                    // Any error or panic cancels peers before scoped joins finish.
                    let stop = control.cancel_on_exit();
                    let gpu = match command {
                        #[cfg(feature = "self_test_support")]
                        Command::SelfTest => GpuContext::for_self_test(ordinal)?,
                        #[allow(unreachable_patterns)]
                        _ => GpuContext::new(ordinal)?,
                    };

                    let result = match command {
                        #[cfg(feature = "p256-public-key")]
                        Command::P256PublicKeyVanity(args) => {
                            modes::p256_public::gpu::run(args, &gpu, stats, control.clone())
                        }
                        #[cfg(feature = "p256-signature")]
                        Command::P256SignatureVanity(args) => {
                            modes::p256_signature::gpu::run(args, &gpu, stats, control.clone())
                        }
                        #[cfg(feature = "rsa-pss")]
                        Command::RsaPssSignatureVanity(args) => {
                            modes::rsa_pss_search::gpu::run(args, &gpu, stats, control.clone())
                        }
                        #[cfg(feature = "rsa-modulus")]
                        Command::RsaModulusVanity(args) => {
                            modes::rsa_modulus::gpu::run(args, &gpu, stats, control.clone())
                        }
                        #[cfg(feature = "solana")]
                        Command::SolanaVanity { prefix, suffix } => modes::solana::gpu::run(
                            ordinal,
                            prefix.clone(),
                            suffix.clone(),
                            &gpu,
                            stats,
                            control.clone(),
                        ),
                        #[cfg(feature = "bitcoin")]
                        Command::BitcoinVanity { prefix, suffix } => modes::bitcoin::gpu::run(
                            ordinal,
                            prefix.clone(),
                            suffix.clone(),
                            &gpu,
                            stats,
                            control.clone(),
                        ),
                        #[cfg(feature = "ethereum")]
                        Command::EthereumVanity { prefix, suffix } => modes::ethereum::gpu::run(
                            ordinal,
                            prefix.clone(),
                            suffix.clone(),
                            &gpu,
                            stats,
                            control.clone(),
                        ),
                        #[cfg(feature = "shallenge")]
                        Command::Shallenge { username, .. } => modes::shallenge::gpu::run(
                            ordinal,
                            username.clone(),
                            shared_best_hash.unwrap(),
                            &gpu,
                            stats,
                            control.clone(),
                        ),
                        #[cfg(feature = "self_test_support")]
                        Command::SelfTest => modes::self_test::gpu::run(ordinal, &gpu),
                    };
                    if result.is_ok() {
                        stop.finish();
                    }
                    result
                }));
            }
            let mut first_error = None;
            for handle in handles {
                let result = handle
                    .join()
                    .unwrap_or_else(|_| Err("CUDA worker panicked".into()));
                if let Err(error) = result {
                    control.cancel();
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
            }
            first_error.map_or(Ok(()), Err)
        })
    }
}
impl Runner for GpuRunner {
    fn device_count(&self) -> usize {
        self.num_devices
    }
    fn run(&self, command: &Command, stats: Arc<GlobalStats>) -> RunResult {
        #[cfg(feature = "crypto-cli")]
        {
            let bounded = match command {
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
            };
            if bounded {
                return crate::common::search_session::run_controlled(
                    stats.clone(),
                    "candidates",
                    |control| {
                        self.run_devices(command, stats.clone(), control.clone())
                            .map_err(|e| e.to_string())?;
                        Ok(control.has_winner())
                    },
                );
            }
        }
        self.run_devices(
            command,
            stats.clone(),
            Arc::new(SearchControl::with_stats(stats)),
        )
    }
}
