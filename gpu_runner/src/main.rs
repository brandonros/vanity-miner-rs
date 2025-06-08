use cust::device::Device;
use cust::module::{Module, ModuleJitOption};
use cust::prelude::Context;
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt;
use cust::memory::CopyDestination;
use cust::{launch, CudaFlags};
use rand::Rng;
use std::error::Error;
use std::sync::{Arc, RwLock};

use common::GlobalStats;

#[derive(Debug, Clone)]
enum Mode {
    Vanity { prefix: String, suffix: String },
    Shallenge { username: String, target_hash: String },
}

// Shared state for the best hash found so far (copied from CPU runner)
struct SharedBestHash {
    hash: [u8; 32],
}

impl SharedBestHash {
    fn new(initial_hash: [u8; 32]) -> Self {
        Self { hash: initial_hash }
    }
    
    fn update_if_better(&mut self, new_hash: [u8; 32]) -> bool {
        // Compare hashes lexicographically (smaller is better)
        if new_hash < self.hash {
            self.hash = new_hash;
            true
        } else {
            false
        }
    }
    
    fn get_current(&self) -> [u8; 32] {
        self.hash
    }
}

fn validate_base58_string(base58_string: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let invalid_characters = ["l", "I", "0", "O"];
    for invalid_character in invalid_characters {
        if base58_string.contains(invalid_character) {
            return Err(format!("base58 string contains invalid character: {}", invalid_character).into());
        }
    }
    Ok(())
}

fn validate_hex_string(hex_string: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    if hex_string.len() != 64 {
        return Err("Hash must be 64 hex characters (32 bytes)".into());
    }
    hex::decode(hex_string).map_err(|e| format!("Invalid hex string: {}", e).into())
}

fn device_main_vanity(
    ordinal: usize, 
    vanity_prefix: String, 
    vanity_suffix: String,
    module: &Module,
    global_stats: Arc<GlobalStats>
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let vanity_prefix_bytes = vanity_prefix.as_bytes();
    let vanity_prefix_len: usize = vanity_prefix_bytes.len();
    let vanity_suffix_bytes = vanity_suffix.as_bytes();
    let vanity_suffix_len: usize = vanity_suffix_bytes.len();
    
    let device = Device::get_device(ordinal as u32)?;
    let ctx = Context::new(device)?;
    cust::context::CurrentContext::set_current(&ctx)?;
    
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let find_vanity_private_key = module.get_function("find_vanity_private_key")?;

    let number_of_streaming_multiprocessors = device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as usize;
    let blocks_per_sm = std::env::var("BLOCKS_PER_SM").unwrap_or("128".to_string()).parse::<usize>().unwrap();
    let threads_per_block = std::env::var("THREADS_PER_BLOCK").unwrap_or("256".to_string()).parse::<usize>().unwrap();
    let blocks_per_grid = number_of_streaming_multiprocessors * blocks_per_sm;
    let operations_per_launch = blocks_per_grid * threads_per_block;

    println!("[{ordinal}] Starting vanity search loop ({} blocks per grid, {} threads per block, {} operations per launch)", blocks_per_grid, threads_per_block, operations_per_launch);

    let mut rng = rand::thread_rng();

    loop {
        let rng_seed: u64 = rng.r#gen::<u64>();
        
        let mut found_matches_slice = [0u32; 1];
        let mut found_private_key = [0u8; 32];
        let mut found_public_key = [0u8; 32];
        let mut found_bs58_encoded_public_key = [0u8; 64];
        let mut found_thread_idx_slice = [0u32; 1];
        
        let vanity_prefix_dev = vanity_prefix_bytes.as_dbuf()?;
        let vanity_suffix_dev = vanity_suffix_bytes.as_dbuf()?;
        let found_matches_slice_dev = found_matches_slice.as_dbuf()?;
        let found_private_key_dev = found_private_key.as_dbuf()?;
        let found_public_key_dev = found_public_key.as_dbuf()?;
        let found_bs58_encoded_public_key_dev = found_bs58_encoded_public_key.as_dbuf()?;    
        let found_thread_idx_slice_dev = found_thread_idx_slice.as_dbuf()?;

        unsafe {
            launch!(
                find_vanity_private_key<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                    vanity_prefix_dev.as_device_ptr(),
                    vanity_prefix_len,
                    vanity_suffix_dev.as_device_ptr(),
                    vanity_suffix_len,
                    rng_seed,
                    found_matches_slice_dev.as_device_ptr(),
                    found_private_key_dev.as_device_ptr(),
                    found_public_key_dev.as_device_ptr(),
                    found_bs58_encoded_public_key_dev.as_device_ptr(),
                    found_thread_idx_slice_dev.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;
        global_stats.add_launch(operations_per_launch);

        found_matches_slice_dev.copy_to(&mut found_matches_slice)?;
        
        let found_matches = found_matches_slice[0];
        if found_matches != 0 {
            found_private_key_dev.copy_to(&mut found_private_key)?;
            found_public_key_dev.copy_to(&mut found_public_key)?;
            found_bs58_encoded_public_key_dev.copy_to(&mut found_bs58_encoded_public_key)?;
            found_thread_idx_slice_dev.copy_to(&mut found_thread_idx_slice)?;

            let found_thread_idx = found_thread_idx_slice[0];
            let found_bs58_encoded_public_key_string = String::from_utf8(found_bs58_encoded_public_key.to_vec()).unwrap();
            println!("[{ordinal}] Vanity match: seed = {rng_seed} thread_idx = {found_thread_idx}");
            println!("[{ordinal}] Vanity match: encoded_public_key = {found_bs58_encoded_public_key_string}");
            println!("[{ordinal}] Vanity match: public_key = {}", hex::encode(found_public_key));
            println!("[{ordinal}] Vanity match: private_key = {}", hex::encode(found_private_key));
            println!("[{ordinal}] Vanity match: wallet = {}", hex::encode([found_private_key, found_public_key].concat()));

            global_stats.add_matches(found_matches as usize);
            global_stats.print_stats(ordinal, found_matches);
        }
    }
}

fn device_main_shallenge(
    ordinal: usize, 
    username: String,
    shared_best_hash: Arc<RwLock<SharedBestHash>>,
    module: &Module,
    global_stats: Arc<GlobalStats>
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let username_bytes = username.as_bytes();
    let username_len: usize = username_bytes.len();
    
    let device = Device::get_device(ordinal as u32)?;
    let ctx = Context::new(device)?;
    cust::context::CurrentContext::set_current(&ctx)?;
    
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let find_better_shallenge_nonce = module.get_function("find_better_shallenge_nonce")?;

    let number_of_streaming_multiprocessors = device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as usize;
    let blocks_per_sm = std::env::var("BLOCKS_PER_SM").unwrap_or("128".to_string()).parse::<usize>().unwrap();
    let threads_per_block = std::env::var("THREADS_PER_BLOCK").unwrap_or("256".to_string()).parse::<usize>().unwrap();
    let blocks_per_grid = number_of_streaming_multiprocessors * blocks_per_sm;
    let operations_per_launch = blocks_per_grid * threads_per_block;

    println!("[{ordinal}] Starting shallenge search loop ({} blocks per grid, {} threads per block, {} operations per launch)", blocks_per_grid, threads_per_block, operations_per_launch);

    let mut rng = rand::thread_rng();

    loop {
        let rng_seed: u64 = rng.r#gen::<u64>();
        
        // Get the current best hash (with minimal lock time)
        let current_target = {
            let best_hash_guard = shared_best_hash.read().unwrap();
            best_hash_guard.get_current()
        };
        
        let mut found_matches_slice = [0u32; 1];
        let mut found_hash = [0u8; 32];
        let mut found_nonce = [0u8; 64];
        let mut found_nonce_len = [0usize; 1];
        let mut found_thread_idx_slice = [0u32; 1];
        
        let username_dev = username_bytes.as_dbuf()?;
        let target_hash_dev = current_target.as_dbuf()?;
        let found_matches_slice_dev = found_matches_slice.as_dbuf()?;
        let found_hash_dev = found_hash.as_dbuf()?;
        let found_nonce_dev = found_nonce.as_dbuf()?;
        let found_nonce_len_dev = found_nonce_len.as_dbuf()?;
        let found_thread_idx_slice_dev = found_thread_idx_slice.as_dbuf()?;

        unsafe {
            launch!(
                find_better_shallenge_nonce<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                    username_dev.as_device_ptr(),
                    username_len,
                    target_hash_dev.as_device_ptr(),
                    rng_seed,
                    found_matches_slice_dev.as_device_ptr(),
                    found_hash_dev.as_device_ptr(),
                    found_nonce_dev.as_device_ptr(),
                    found_nonce_len_dev.as_device_ptr(),
                    found_thread_idx_slice_dev.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;
        global_stats.add_launch(operations_per_launch);

        found_matches_slice_dev.copy_to(&mut found_matches_slice)?;
        
        let found_matches = found_matches_slice[0];
        if found_matches != 0 {
            found_hash_dev.copy_to(&mut found_hash)?;
            found_nonce_dev.copy_to(&mut found_nonce)?;
            found_nonce_len_dev.copy_to(&mut found_nonce_len)?;
            found_thread_idx_slice_dev.copy_to(&mut found_thread_idx_slice)?;

            let found_thread_idx = found_thread_idx_slice[0];
            let nonce_len = found_nonce_len[0];
            let nonce_string = String::from_utf8(found_nonce[..nonce_len].to_vec()).unwrap();
            
            // Try to update the global best hash
            let was_global_best = {
                let mut best_hash_guard = shared_best_hash.write().unwrap();
                best_hash_guard.update_if_better(found_hash)
            };
            
            if was_global_best {
                println!("[{ordinal}] NEW GLOBAL BEST found: seed = {rng_seed} thread_idx = {found_thread_idx}");
                println!("[{ordinal}] NEW GLOBAL BEST hash: {}", hex::encode(found_hash));
                println!("[{ordinal}] NEW GLOBAL BEST nonce: {}", nonce_string);
                println!("[{ordinal}] Challenge string: {}/{}", username, nonce_string);

                global_stats.add_matches(found_matches as usize);
                global_stats.print_stats(ordinal, found_matches);
            }
        }
    }
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
    let ptx_path = std::path::Path::new("kernels.ptx");
    let ptx = std::fs::read_to_string(ptx_path)
        .map_err(|e| format!("Failed to read PTX file: {}", e))?;
    let module = Module::from_ptx(ptx, &[
        ModuleJitOption::MaxRegisters(256),
    ])?;

    match mode {
        Mode::Vanity { prefix, suffix } => {
            device_main_vanity(ordinal, prefix, suffix, &module, global_stats)
        }
        Mode::Shallenge { username, .. } => {
            let shared_best_hash = shared_best_hash.expect("SharedBestHash required for shallenge mode");
            device_main_shallenge(ordinal, username, shared_best_hash, &module, global_stats)
        }
    }
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = std::env::args().collect::<Vec<String>>();
    
    let mode = if args.len() == 4 && args[1] == "vanity" {
        let vanity_prefix = args[2].clone();
        let vanity_suffix = args[3].clone();
        validate_base58_string(&vanity_prefix)?;
        validate_base58_string(&vanity_suffix)?;
        Mode::Vanity { prefix: vanity_prefix, suffix: vanity_suffix }
    } else if args.len() == 4 && args[1] == "shallenge" {
        let username = args[2].clone();
        let target_hash = args[3].clone();
        Mode::Shallenge { username, target_hash }
    } else {
        println!("Usage:");
        println!("  {} vanity <prefix> <suffix>", args[0]);
        println!("  {} shallenge <username> <target_hash_hex>", args[0]);
        std::process::exit(1);
    };

    cust::init(CudaFlags::empty())?;
    let num_devices = Device::num_devices()?;
    println!("Found {} CUDA devices", num_devices);

    let global_stats = match &mode {
        Mode::Vanity { prefix, suffix } => Arc::new(GlobalStats::new(
            num_devices as usize, 
            prefix.len(),
            suffix.len()
        )),
        Mode::Shallenge { username, .. } => Arc::new(GlobalStats::new(
            num_devices as usize, 
            username.len(),
            0 // No suffix for shallenge
        )),
    };

    // Create shared state for shallenge mode
    let shared_best_hash = match &mode {
        Mode::Vanity { .. } => None,
        Mode::Shallenge { target_hash, .. } => {
            let target_hash_bytes = validate_hex_string(target_hash)?;
            let mut initial_target = [0u8; 32];
            initial_target.copy_from_slice(&target_hash_bytes);
            Some(Arc::new(RwLock::new(SharedBestHash::new(initial_target))))
        }
    };

    match &mode {
        Mode::Vanity { prefix, suffix } => {
            println!("Searching for vanity key with prefix '{}' and suffix '{}'", prefix, suffix);
        }
        Mode::Shallenge { username, target_hash } => {
            println!("Starting shallenge for username '{}' with target hash '{}'", username, target_hash);
        }
    }

    let mut handles = Vec::new();
    for i in 0..num_devices as usize {
        println!("Starting device {}", i);
        let mode_clone = mode.clone();
        let shared_best_hash_clone = shared_best_hash.clone();
        let stats_clone = Arc::clone(&global_stats);
        handles.push(std::thread::spawn(move || device_main(
            i,
            mode_clone,
            shared_best_hash_clone,
            stats_clone
        )));
    }

    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    Ok(())
}