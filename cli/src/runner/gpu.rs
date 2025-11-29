use crate::args::Command;
use crate::modes;
use crate::runner::Runner;
use backtrace::Backtrace;
use common::{GlobalStats, SharedBestHash};
use cust::device::Device;
use cust::module::{Module, ModuleJitOption};
use cust::prelude::Context;
use cust::CudaFlags;
use std::error::Error;
use std::sync::{Arc, RwLock};

pub struct GpuRunner {
    num_devices: usize,
}

impl GpuRunner {
    pub fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        cust::init(CudaFlags::empty())?;
        let num_devices = Device::num_devices()? as usize;
        println!("Found {} CUDA devices", num_devices);
        Ok(Self { num_devices })
    }

    fn load_module(ordinal: usize) -> Result<Module, Box<dyn Error + Send + Sync>> {
        let device = Device::get_device(ordinal as u32)?;
        let ctx = Context::new(device)?;
        cust::context::CurrentContext::set_current(&ctx)?;

        println!("[{ordinal}] Loading module...");
        let module = {
            let cubin_path = std::env::var("CUBIN_PATH");
            let ptx_path = std::env::var("PTX_PATH");
            if let Ok(cubin_path) = cubin_path {
                let cubin = std::fs::read(cubin_path)
                    .map_err(|e| format!("Failed to read CUBIN file: {}", e))?;
                Module::from_cubin(cubin, &[])?
            } else if let Ok(ptx_path) = ptx_path {
                let ptx = std::fs::read_to_string(ptx_path)
                    .map_err(|e| format!("Failed to read PTX file: {}", e))?;
                Module::from_ptx(ptx, &[ModuleJitOption::MaxRegisters(256)])?
            } else {
                return Err("CUBIN_PATH or PTX_PATH environment variable is required".into());
            }
        };
        println!("[{ordinal}] Module loaded");
        Ok(module)
    }
}

impl Runner for GpuRunner {
    fn device_count(&self) -> usize {
        self.num_devices
    }

    fn run(&self, command: &Command, stats: Arc<GlobalStats>) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Set up panic hook for better error reporting
        std::panic::set_hook(Box::new(|panic_info| {
            let backtrace = Backtrace::new();
            eprintln!("Thread panicked: {}", panic_info);
            eprintln!("Backtrace:\n{:?}", backtrace);
        }));

        // Create shared state for shallenge mode
        let shared_best_hash: Option<Arc<RwLock<SharedBestHash>>> = match command {
            Command::Shallenge { target_hash, .. } => {
                let target_hash_bytes = hex::decode(target_hash)?;
                let mut initial_target = [0u8; 32];
                initial_target.copy_from_slice(&target_hash_bytes);
                Some(Arc::new(RwLock::new(SharedBestHash::new(initial_target))))
            }
            _ => None,
        };

        // Spawn device threads
        let mut handles = Vec::new();
        for i in 0..self.num_devices {
            println!("Starting device {}", i);
            let command_clone = command.clone();
            let shared_best_hash_clone = shared_best_hash.clone();
            let stats_clone = Arc::clone(&stats);

            handles.push(std::thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
                let module = Self::load_module(i)?;

                match command_clone {
                    Command::SolanaVanity { prefix, suffix } => {
                        modes::solana::gpu::run(i, prefix, suffix, &module, stats_clone)
                    }
                    Command::BitcoinVanity { prefix, suffix } => {
                        modes::bitcoin::gpu::run(i, prefix, suffix, &module, stats_clone)
                    }
                    Command::EthereumVanity { prefix, suffix } => {
                        modes::ethereum::gpu::run(i, prefix, suffix, &module, stats_clone)
                    }
                    Command::Shallenge { username, .. } => {
                        let shared = shared_best_hash_clone.expect("SharedBestHash required for shallenge mode");
                        modes::shallenge::gpu::run(i, username, shared, &module, stats_clone)
                    }
                }.map_err(|e| {
                    let bt = Backtrace::new();
                    eprintln!("Error in device {}: {}", i, e);
                    eprintln!("Backtrace:\n{:?}", bt);
                    e
                })
            }));
        }

        // Wait for threads
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
