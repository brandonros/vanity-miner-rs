use crate::args::Command;
use crate::common::{GlobalStats, SharedBestHash};
use crate::modes;
use crate::runner::Runner;
use cuda_core::{CudaContext, IntoResult, sys::cuDeviceGetCount};
use kernels::kernels::LoadedModule;
use std::error::Error;
use std::ffi::c_int;
use std::mem::MaybeUninit;
use std::sync::{Arc, RwLock};

pub struct GpuRunner {
    num_devices: usize,
}

impl GpuRunner {
    pub fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let num_devices = unsafe {
            let mut count = MaybeUninit::<c_int>::uninit();
            cuDeviceGetCount(count.as_mut_ptr()).result()?;
            count.assume_init() as usize
        };
        println!("Found {} CUDA devices", num_devices);
        Ok(Self { num_devices })
    }

    fn load_module(
        ordinal: usize,
    ) -> Result<(Arc<CudaContext>, LoadedModule), Box<dyn Error + Send + Sync>> {
        let ctx = CudaContext::new(ordinal)?;
        println!("[{ordinal}] Loading module...");
        let module = kernels::kernels::load(&ctx)?;
        println!("[{ordinal}] Module loaded");
        Ok((ctx, module))
    }
}

impl Runner for GpuRunner {
    fn device_count(&self) -> usize {
        self.num_devices
    }

    fn run(
        &self,
        command: &Command,
        stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let shared_best_hash: Option<Arc<RwLock<SharedBestHash>>> = match command {
            Command::Shallenge { target_hash, .. } => {
                let target_hash_bytes = hex::decode(target_hash)?;
                let mut initial_target = [0u8; 32];
                initial_target.copy_from_slice(&target_hash_bytes);
                Some(Arc::new(RwLock::new(SharedBestHash::new(initial_target))))
            }
            _ => None,
        };

        let mut handles = Vec::new();
        for i in 0..self.num_devices {
            println!("Starting device {}", i);
            let command_clone = command.clone();
            let shared_best_hash_clone = shared_best_hash.clone();
            let stats_clone = Arc::clone(&stats);

            handles.push(std::thread::spawn(
                move || -> Result<(), Box<dyn Error + Send + Sync>> {
                    // The Arc<CudaContext> must outlive the module — when the last
                    // ref drops, cuDevicePrimaryCtxRelease invalidates every
                    // CUmodule under it.
                    let (ctx, module) = Self::load_module(i)?;

                    match command_clone {
                        Command::SolanaVanity { prefix, suffix } => {
                            modes::solana::gpu::run(i, prefix, suffix, &ctx, &module, stats_clone)
                        }
                        Command::BitcoinVanity { prefix, suffix } => {
                            modes::bitcoin::gpu::run(i, prefix, suffix, &ctx, &module, stats_clone)
                        }
                        Command::EthereumVanity { prefix, suffix } => {
                            modes::ethereum::gpu::run(i, prefix, suffix, &ctx, &module, stats_clone)
                        }
                        Command::Shallenge { username, .. } => {
                            let shared = shared_best_hash_clone
                                .expect("SharedBestHash required for shallenge mode");
                            modes::shallenge::gpu::run(
                                i,
                                username,
                                shared,
                                &ctx,
                                &module,
                                stats_clone,
                            )
                        }
                    }
                    .inspect_err(|e| eprintln!("Error in device {}: {}", i, e))
                },
            ));
        }

        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(result) => {
                    if let Err(e) = result {
                        eprintln!("Device {} returned error: {}", i, e);
                        return Err(e);
                    }
                }
                Err(panic_payload) => {
                    eprintln!("Device {} thread panicked!", i);
                    let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic".to_string()
                    };
                    eprintln!("Panic message: {}", panic_msg);
                    return Err(format!("Device {} panicked: {}", i, panic_msg).into());
                }
            }
        }

        Ok(())
    }
}
