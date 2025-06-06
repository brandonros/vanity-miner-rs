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

use common::GlobalStats;

fn validate_base58_string(base58_string: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let invalid_characters = [
        "l", // lowercase L
        "I", // uppercase I
        "0", // zero
        "O" // uppercase O
    ];
    for invalid_character in invalid_characters {
        if base58_string.contains(invalid_character) == true {
            return Err(format!("base58 string contains invalid character: {}", invalid_character).into());
        }
    }
    Ok(())
}

fn device_main(
    ordinal: usize, 
    vanity_prefix: String, 
    vanity_suffix: String,
    global_stats: Arc<GlobalStats>
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // convert the vanity prefix to a byte array
    let vanity_prefix_bytes = vanity_prefix.as_bytes();
    let vanity_prefix_len: usize = vanity_prefix_bytes.len();

    // convert the vanity suffix to a byte array
    let vanity_suffix_bytes = vanity_suffix.as_bytes();
    let vanity_suffix_len: usize = vanity_suffix_bytes.len();
    
    let device = Device::get_device(ordinal as u32)?;
    let ctx = Context::new(device)?;
    cust::context::CurrentContext::set_current(&ctx)?;

    // Load the pre-compiled PTX that was generated during build
    println!("[{ordinal}] Loading module...");
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));
    let module = Module::from_ptx(ptx, &[
        ModuleJitOption::MaxRegisters(256),
    ])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let find_vanity_private_key = module.get_function("find_vanity_private_key")?;

    // get the number of streaming multiprocessors
    let number_of_streaming_multiprocessors = device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as usize;
    let blocks_per_sm = 128;
    let threads_per_block = 256;
    let blocks_per_grid = number_of_streaming_multiprocessors * blocks_per_sm;
    let operations_per_launch = blocks_per_grid * threads_per_block;

    println!("[{ordinal}] Starting search loop ({} blocks per grid, {} threads per block, {} operations per launch)", blocks_per_grid, threads_per_block, operations_per_launch);

    let mut rng = rand::thread_rng();

    loop {
        let rng_seed: u64 = rng.r#gen::<u64>();
        
        let mut found_matches_slice = [0.0f32; 1];
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

        // Launch the kernel
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                find_vanity_private_key<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                    // input
                    vanity_prefix_dev.as_device_ptr(),
                    vanity_prefix_len,
                    vanity_suffix_dev.as_device_ptr(),
                    vanity_suffix_len,
                    rng_seed,
                    // output
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

            // Format results
            let found_thread_idx = found_thread_idx_slice[0];
            let found_bs58_encoded_public_key_string = String::from_utf8(found_bs58_encoded_public_key.to_vec()).unwrap();
            println!("[{ordinal}] First match: seed = {rng_seed} thread_idx = {found_thread_idx}");
            println!("[{ordinal}] First match: encoded_public_key = {found_bs58_encoded_public_key_string}");
            println!("[{ordinal}] First match: public_key = {}", hex::encode(found_public_key));
            println!("[{ordinal}] First match: private_key = {}", hex::encode(found_private_key));
            println!("[{ordinal}] First match: wallet = {}", hex::encode([found_private_key, found_public_key].concat()));

            // Print stats using global counters
            global_stats.add_matches(found_matches as usize);
            global_stats.print_stats(
                ordinal,
                found_matches,
            );
        }
    }
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Define the vanity prefix we're looking for
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() != 3 {
        println!("Usage: {} <vanity_prefix> <vanity_suffix>", args[0]);
        std::process::exit(1);
    }
    let vanity_prefix = args[1].to_string();
    let vanity_suffix = args[2].to_string();    

    // check if the vanity prefix or suffix contains any of the forbidden characters
    validate_base58_string(&vanity_prefix)?;
    validate_base58_string(&vanity_suffix)?;

    // Initialize CUDA context and get default stream
    cust::init(CudaFlags::empty())?;
    let num_devices = Device::num_devices()?;
    println!("Found {} CUDA devices", num_devices);

    // Initialize global stats
    let global_stats = Arc::new(GlobalStats::new(
        num_devices as usize, 
        vanity_prefix.len(),
        vanity_suffix.len()
    ));

    // start device threads
    let mut handles = Vec::new();
    for i in 0..num_devices as usize {
        println!("Starting device {}", i);
        let vanity_prefix_clone = vanity_prefix.clone();
        let vanity_suffix_clone = vanity_suffix.clone();
        let stats_clone = Arc::clone(&global_stats);
        handles.push(std::thread::spawn(move || device_main(
            i,
            vanity_prefix_clone,
            vanity_suffix_clone,
            stats_clone
        )));
    }

    // wait for all device threads to finish
    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    Ok(())
}
