mod stats;

use cust::device::Device;
use cust::module::{Module, ModuleJitOption};
use cust::prelude::Context;
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt;
use cust::memory::CopyDestination;
use cust::{launch, CudaFlags};
use rand::Rng;
use std::error::Error;
use std::sync::Arc;

use crate::stats::GlobalStats;

fn device_main(
    ordinal: usize, 
    vanity_prefix: String, 
    blocks_per_grid: usize, 
    threads_per_block: usize,
    global_stats: Arc<GlobalStats>
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // check if the vanity prefix contains any of the forbidden characters
    assert!(vanity_prefix.contains("l") == false); // lowercase L
    assert!(vanity_prefix.contains("I") == false); // uppercase i
    assert!(vanity_prefix.contains("0") == false); // zero
    assert!(vanity_prefix.contains("O") == false); // uppercase o

    // convert the vanity prefix to a byte array
    let vanity_prefix_bytes = vanity_prefix.as_bytes();
    let vanity_prefix_len: usize = vanity_prefix_bytes.len();
    
    let device = Device::get_device(ordinal as u32)?;
    let ctx = Context::new(device)?;
    cust::context::CurrentContext::set_current(&ctx)?;

    // Load the pre-compiled PTX that was generated during build
    println!("[{ordinal}] Loading module...");
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));
    let module = Module::from_ptx(ptx, &[
        ModuleJitOption::MaxRegisters(256)
    ])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let find_vanity_private_key = module.get_function("find_vanity_private_key")?;

    let operations_per_launch = blocks_per_grid * threads_per_block;
    println!("[{ordinal}] Starting search loop...");

    let mut rng = rand::thread_rng();

    loop {
        let rng_seed: u64 = rng.r#gen::<u64>();
        
        let mut found_matches_slice = [0.0f32; 1];
        let mut found_private_key = [0u8; 32];
        let mut found_public_key = [0u8; 32];
        let mut found_bs58_encoded_public_key = [0u8; 64];
        let mut found_thread_idx_slice = [0u32; 1];
        
        let vanity_prefix_dev = vanity_prefix_bytes.as_dbuf()?;
        let found_matches_slice_dev = found_matches_slice.as_dbuf()?;
        let found_private_key_dev = found_private_key.as_dbuf()?;
        let found_public_key_dev = found_public_key.as_dbuf()?;
        let found_bs58_encoded_public_key_dev = found_bs58_encoded_public_key.as_dbuf()?;    
        let found_thread_idx_slice_dev = found_thread_idx_slice.as_dbuf()?;

        // Launch the kernel
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                find_vanity_private_key<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                    vanity_prefix_dev.as_device_ptr(),
                    vanity_prefix_len,
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

        // increment stats
        global_stats.add_launch(operations_per_launch);

        // Check if we found a match
        found_matches_slice_dev.copy_to(&mut found_matches_slice)?;
        
        let found_matches = found_matches_slice[0];
        if found_matches != 0.0 {
            // We found a match! Copy results back to host
            found_private_key_dev.copy_to(&mut found_private_key)?;
            found_public_key_dev.copy_to(&mut found_public_key)?;
            found_bs58_encoded_public_key_dev.copy_to(&mut found_bs58_encoded_public_key)?;
            found_thread_idx_slice_dev.copy_to(&mut found_thread_idx_slice)?;

            global_stats.add_matches(found_matches as usize);

            // Format results
            let found_thread_idx = found_thread_idx_slice[0];
            let wallet_formatted_result = hex::encode([found_private_key, found_public_key].concat());
            let encoded_public_key_string = String::from_utf8(found_bs58_encoded_public_key.to_vec()).unwrap();
            println!("[{ordinal}] First match: seed = {rng_seed} thread_idx = {found_thread_idx} encoded_public_key = {encoded_public_key_string}");
            println!("[{ordinal}] First match: public_key = {} private_key = {} wallet = {}", hex::encode(found_public_key), hex::encode(found_private_key), wallet_formatted_result);

            // Print stats using global counters
            global_stats.print_stats(
                ordinal,
                found_matches,
            );
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Define the vanity prefix we're looking for
    let args = std::env::args().collect::<Vec<String>>();
    let vanity_prefix = args[1].to_string();
    let blocks_per_grid = args[2].parse::<usize>().unwrap();
    let threads_per_block = args[3].parse::<usize>().unwrap();

    // Initialize CUDA context and get default stream
    cust::init(CudaFlags::empty())?;
    let num_devices = Device::num_devices()?;
    println!("Found {} CUDA devices", num_devices);

    // Initialize global stats
    let global_stats = Arc::new(GlobalStats::new());

    // start threads
    let mut handles = Vec::new();
    for i in 0..num_devices as usize {
        println!("Starting device {}", i);
        let vanity_prefix_clone = vanity_prefix.clone();
        let stats_clone = Arc::clone(&global_stats);
        let handle = std::thread::spawn(move || device_main(
            i,
            vanity_prefix_clone,
            blocks_per_grid,
            threads_per_block,
            stats_clone
        ));
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    Ok(())
}
