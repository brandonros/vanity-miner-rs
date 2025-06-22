mod add;
mod solana;
mod bitcoin;
mod shallenge;
mod ethereum;

use cust::device::Device;
use cust::module::{Module, ModuleJitOption};
use cust::prelude::Context;
use cust::CudaFlags;
use backtrace::Backtrace;

use std::error::Error;
use std::sync::{Arc, RwLock};

use common::GlobalStats;
use common::SharedBestHash;

#[derive(Debug, Clone)]
enum Mode {
    Add,
    SolanaVanity { prefix: String, suffix: String },
    BitcoinVanity { prefix: String, suffix: String },
    EthereumVanity { prefix: String, suffix: String },
    Shallenge { username: String, target_hash: String },
}

fn device_main(
    ordinal: usize, 
    mode: Mode,
    shared_best_hash: Option<Arc<RwLock<SharedBestHash>>>,
    global_stats: Arc<GlobalStats>
) -> Result<(), Box<dyn Error + Send + Sync>> {
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
            Module::from_ptx(ptx, &[
                ModuleJitOption::MaxRegisters(256),
            ])
        } else {
            return Err("CUBIN_PATH or PTX_PATH environment variable is required".into());
        }
    };
    println!("[{ordinal}] Module loaded");

    match mode {
        Mode::Add => {
            add::device_main_add(ordinal, &module)
        }
        Mode::SolanaVanity { prefix, suffix } => {
            solana::device_main_solana_vanity(ordinal, prefix, suffix, &module, global_stats)
        }
        Mode::BitcoinVanity { prefix, suffix } => {
            bitcoin::device_main_bitcoin_vanity(ordinal, prefix, suffix, &module, global_stats)
        }
        Mode::EthereumVanity { prefix, suffix } => {
            ethereum::device_main_ethereum_vanity(ordinal, prefix, suffix, &module, global_stats)
        }
        Mode::Shallenge { username, .. } => {
            let shared_best_hash = shared_best_hash.expect("SharedBestHash required for shallenge mode");
            shallenge::device_main_shallenge(ordinal, username, shared_best_hash, &module, global_stats)
        }
    }
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    std::panic::set_hook(Box::new(|panic_info| {
        let backtrace = Backtrace::new();
        eprintln!("Thread panicked: {}", panic_info);
        eprintln!("Backtrace:\n{:?}", backtrace);
    }));

    let args = std::env::args().collect::<Vec<String>>();
    
    let mode = if args.len() == 2 && args[1] == "add" {
        Mode::Add
    } else if args.len() == 4 && args[1] == "solana-vanity" {
        let vanity_prefix = args[2].clone();
        let vanity_suffix = args[3].clone();
        if vanity_prefix.len() > 0 {
            common::validate_base58_string(&vanity_prefix)?;
        }
        if vanity_suffix.len() > 0 {
            common::validate_base58_string(&vanity_suffix)?;
        }
        Mode::SolanaVanity { prefix: vanity_prefix, suffix: vanity_suffix }
    } else if args.len() == 4 && args[1] == "bitcoin-vanity" {
        let vanity_prefix = args[2].clone();
        let vanity_suffix = args[3].clone();
        if vanity_prefix.len() > 0 {
            common::validate_bech32_string(&vanity_prefix)?;
        }
        if vanity_suffix.len() > 0 {
            common::validate_bech32_string(&vanity_suffix)?;
        }
        Mode::BitcoinVanity { prefix: vanity_prefix, suffix: vanity_suffix }
    } else if args.len() == 4 && args[1] == "ethereum-vanity" {
        let vanity_prefix = args[2].clone();
        let vanity_suffix = args[3].clone();
        if vanity_prefix.len() > 0 {
            common::validate_hex_string(&vanity_prefix)?;
        }
        if vanity_suffix.len() > 0 {
            common::validate_hex_string(&vanity_suffix)?;
        }
        Mode::EthereumVanity { prefix: vanity_prefix, suffix: vanity_suffix }
    } else if args.len() == 4 && args[1] == "shallenge" {
        let username = args[2].clone();
        let target_hash = args[3].clone();
        common::validate_hex_string(&target_hash)?;
        Mode::Shallenge { username, target_hash }
    } else {
        println!("Usage:");
        println!("  {} add", args[0]);
        println!("  {} solana-vanity <prefix> <suffix>", args[0]);
        println!("  {} bitcoin-vanity <prefix> <suffix>", args[0]);
        println!("  {} ethereum-vanity <prefix> <suffix>", args[0]);
        println!("  {} shallenge <username> <target_hash_hex>", args[0]);
        std::process::exit(1);
    };

    cust::init(CudaFlags::empty())?;
    let num_devices = Device::num_devices()?;
    println!("Found {} CUDA devices", num_devices);

    let global_stats = match &mode {
        Mode::SolanaVanity { prefix, suffix } => Arc::new(GlobalStats::new(
            num_devices as usize, 
            prefix.len(),
            suffix.len()
        )),
        Mode::BitcoinVanity { prefix, suffix } => Arc::new(GlobalStats::new(
            num_devices as usize, 
            prefix.len(),
            suffix.len()
        )),
        Mode::EthereumVanity { prefix, suffix } => Arc::new(GlobalStats::new(
            num_devices as usize,
            prefix.len(),
            suffix.len()
        )),
        Mode::Shallenge { username, .. } => Arc::new(GlobalStats::new(
            num_devices as usize, 
            username.len(),
            0 // No suffix for shallenge
        )),
        Mode::Add => Arc::new(GlobalStats::new(
            num_devices as usize, 
            0, // No prefix for add
            0 // No suffix for add
        )),
    };

    // Create shared state for shallenge mode
    let shared_best_hash = match &mode {
        Mode::Add => None,
        Mode::SolanaVanity { .. } => None,
        Mode::BitcoinVanity { .. } => None,
        Mode::EthereumVanity { .. } => None,
        Mode::Shallenge { target_hash, .. } => {
            let target_hash_bytes = hex::decode(target_hash)?;
            let mut initial_target = [0u8; 32];
            initial_target.copy_from_slice(&target_hash_bytes);
            Some(Arc::new(RwLock::new(SharedBestHash::new(initial_target))))
        }
    };

    // log
    match &mode {
        Mode::Add => {
            println!("Running add mode");
        }
        Mode::SolanaVanity { prefix, suffix } => {
            println!("Searching for solana vanity key with prefix '{}' and suffix '{}'", prefix, suffix);
        }
        Mode::BitcoinVanity { prefix, suffix } => {
            println!("Searching for bitcoin vanity key with prefix '{}' and suffix '{}'", prefix, suffix);
        }
        Mode::EthereumVanity { prefix, suffix } => {
            println!("Searching for ethereum vanity key with prefix '{}' and suffix '{}'", prefix, suffix);
        }
        Mode::Shallenge { username, target_hash } => {
            println!("Starting shallenge for username '{}' with target hash '{}'", username, target_hash);
        }
    }

    // spawn threads
    let mut handles = Vec::new();
    for i in 0..num_devices as usize {
        println!("Starting device {}", i);
        let mode_clone = mode.clone();
        let shared_best_hash_clone = shared_best_hash.clone();
        let stats_clone = Arc::clone(&global_stats);
        handles.push(std::thread::spawn(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            // Wrap the device_main call to capture any errors with context
            device_main(i, mode_clone, shared_best_hash_clone, stats_clone)
                .map_err(|e| {
                    let bt = Backtrace::new();
                    eprintln!("Error in device {}: {}", i, e);
                    eprintln!("Backtrace:\n{:?}", bt);
                    e
                })
        }));
    }

    // wait for threads
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
                
                // Try to extract panic message
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
