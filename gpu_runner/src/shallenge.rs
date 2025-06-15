use cust::device::Device;
use cust::module::Module;
use cust::prelude::Context;
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt;
use cust::memory::CopyDestination;
use cust::launch;
use rand::Rng;
use std::error::Error;
use std::sync::{Arc, RwLock};

use common::GlobalStats;
use common::SharedBestHash;

pub fn device_main_shallenge(
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