use cust::device::Device;
use cust::module::Module;
use cust::prelude::Context;
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt;
use cust::memory::CopyDestination;
use cust::launch;
use rand::Rng;
use std::error::Error;
use std::sync::Arc;

use common::GlobalStats;

pub fn device_main_solana_vanity(
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
    let find_solana_vanity_private_key = module.get_function("find_solana_vanity_private_key")?;

    let number_of_streaming_multiprocessors = device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as usize;
    let blocks_per_sm = std::env::var("BLOCKS_PER_SM").unwrap_or("128".to_string()).parse::<usize>().unwrap();
    let threads_per_block = std::env::var("THREADS_PER_BLOCK").unwrap_or("256".to_string()).parse::<usize>().unwrap();
    let blocks_per_grid = number_of_streaming_multiprocessors * blocks_per_sm;
    let operations_per_launch = blocks_per_grid * threads_per_block;

    println!("[{ordinal}] Starting vanity search loop ({} blocks per grid, {} threads per block, {} operations per launch)", blocks_per_grid, threads_per_block, operations_per_launch);

    let mut rng = rand::thread_rng();

    loop {
        let rng_seed: u64 = rng.r#gen::<u64>();
        
        let mut found_matches_slice = [0.0f32; 1];
        let mut found_private_key = [0u8; 32];
        let mut found_public_key = [0u8; 32];
        let mut found_encoded_public_key = [0u8; 64];
        let mut found_thread_idx_slice = [0u32; 1];
        
        let vanity_prefix_dev = vanity_prefix_bytes.as_dbuf()?;
        let vanity_suffix_dev = vanity_suffix_bytes.as_dbuf()?;
        let found_matches_slice_dev = found_matches_slice.as_dbuf()?;
        let found_private_key_dev = found_private_key.as_dbuf()?;
        let found_public_key_dev = found_public_key.as_dbuf()?;
        let found_encoded_public_key_dev = found_encoded_public_key.as_dbuf()?;    
        let found_thread_idx_slice_dev = found_thread_idx_slice.as_dbuf()?;

        unsafe {
            launch!(
                find_solana_vanity_private_key<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                    vanity_prefix_dev.as_device_ptr(),
                    vanity_prefix_len,
                    vanity_suffix_dev.as_device_ptr(),
                    vanity_suffix_len,
                    rng_seed,
                    found_matches_slice_dev.as_device_ptr(),
                    found_private_key_dev.as_device_ptr(),
                    found_public_key_dev.as_device_ptr(),
                    found_encoded_public_key_dev.as_device_ptr(),
                    found_thread_idx_slice_dev.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;
        global_stats.add_launch(operations_per_launch);

        found_matches_slice_dev.copy_to(&mut found_matches_slice)?;
        
        let found_matches = found_matches_slice[0];
        if found_matches != 0.0 {
            found_private_key_dev.copy_to(&mut found_private_key)?;
            found_public_key_dev.copy_to(&mut found_public_key)?;
            found_encoded_public_key_dev.copy_to(&mut found_encoded_public_key)?;
            found_thread_idx_slice_dev.copy_to(&mut found_thread_idx_slice)?;

            let found_thread_idx = found_thread_idx_slice[0];
            let found_encoded_public_key_string = String::from_utf8(found_encoded_public_key.to_vec()).unwrap();
            println!("[{ordinal}] Vanity match: seed = {rng_seed} thread_idx = {found_thread_idx}");
            println!("[{ordinal}] Vanity match: encoded_public_key = {found_encoded_public_key_string}");
            println!("[{ordinal}] Vanity match: public_key = {}", hex::encode(found_public_key));
            println!("[{ordinal}] Vanity match: private_key = {}", hex::encode(found_private_key));
            println!("[{ordinal}] Vanity match: wallet = {}", hex::encode([found_private_key, found_public_key].concat()));

            global_stats.add_matches(found_matches as usize);
            global_stats.print_stats(ordinal, found_matches as u32);
        }
    }
}