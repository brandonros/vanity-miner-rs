use cust::device::Device;
use cust::module::Module;
use cust::prelude::Context;
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt;
use cust::memory::CopyDestination;
use cust::{launch, CudaFlags};
use rand::Rng;
use std::time::Instant;
use std::error::Error;

fn device_main(ordinal: usize, vanity_prefix: String, blocks_per_grid: usize, threads_per_block: usize) -> Result<(), Box<dyn Error + Send + Sync>> {
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

    // Load the pre-compiled PTX that was generated during build
    println!("[{ordinal}] Loading module...");
    let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));
    let module = Module::from_ptx(ptx, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let find_vanity_private_key = module.get_function("find_vanity_private_key")?;

    let mut launches = 0;
    let operations_per_launch = blocks_per_grid * threads_per_block;
    println!("[{ordinal}] Starting search loop...");

    let mut rng = rand::thread_rng();
    let start_time = Instant::now();
    let mut matches_found = 0;
    let mut total_operations = 0;

    loop {
        launches += 1;
        total_operations += operations_per_launch;
        let rng_seed: u64 = rng.r#gen::<u64>();
        
        let mut found_flag = [0.0f32; 1];
        let mut found_private_key = [0u8; 32];
        let mut found_public_key = [0u8; 32];
        let mut found_bs58_encoded_public_key = [0u8; 44];
        
        let vanity_prefix_dev = vanity_prefix_bytes.as_dbuf()?;
        let found_flag_dev = found_flag.as_dbuf()?;
        let found_private_key_dev = found_private_key.as_dbuf()?;
        let found_public_key_dev = found_public_key.as_dbuf()?;
        let found_bs58_encoded_public_key_dev = found_bs58_encoded_public_key.as_dbuf()?;    
        
        // Launch the kernel
        unsafe {
            launch!(
                // slices are passed as two parameters, the pointer and the length.
                find_vanity_private_key<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                    vanity_prefix_dev.as_device_ptr(),
                    vanity_prefix_len,
                    rng_seed,
                    found_flag_dev.as_device_ptr(),
                    found_private_key_dev.as_device_ptr(),
                    found_public_key_dev.as_device_ptr(),
                    found_bs58_encoded_public_key_dev.as_device_ptr(),
                )
            )?;
        }

        stream.synchronize()?;

        // Check if we found a match
        found_flag_dev.copy_to(&mut found_flag)?;
        
        if found_flag[0] != 0.0 {
            // We found a match! Copy results back to host
            found_private_key_dev.copy_to(&mut found_private_key)?;
            found_public_key_dev.copy_to(&mut found_public_key)?;
            found_bs58_encoded_public_key_dev.copy_to(&mut found_bs58_encoded_public_key)?;

            // print
            let wallet_formatted_result = hex::encode([found_private_key, found_public_key].concat());
            let public_key_string = String::from_utf8(found_bs58_encoded_public_key.to_vec()).unwrap();
            println!("[{ordinal}] Found match: {public_key_string} {wallet_formatted_result}");

            // increment stats
            matches_found += found_flag[0] as usize;
            let elapsed = start_time.elapsed();
            let elapsed_seconds = elapsed.as_secs_f64();
            let launches_per_second = launches as f64 / elapsed_seconds;
            let operations_per_second = total_operations as f64 / elapsed_seconds / 1_000_000.00;
            let matches_per_second = matches_found as f64 / elapsed_seconds;
            println!("[{ordinal}] Found {matches_found} matches in {elapsed_seconds:.2}s ({matches_per_second:.6} matches/sec, {launches_per_second:.2} launches/sec, {operations_per_second:.2}M ops/sec) with {launches} launches {total_operations} operations {operations_per_launch} ops/launch {blocks_per_grid} blocks/grid {threads_per_block} threads/block");
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

    let mut handles = Vec::new();
    for i in 0..num_devices as usize {
        println!("Starting device {}", i);
        let vanity_prefix_clone = vanity_prefix.clone();
        let handle = std::thread::spawn(move || device_main(i, vanity_prefix_clone, blocks_per_grid, threads_per_block));
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    Ok(())
}
