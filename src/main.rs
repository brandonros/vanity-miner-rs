use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use rand::Rng;
use std::time::Instant;

fn device_main(ordinal: usize, vanity_prefix: String) -> Result<(), DriverError> {
    // check if the vanity prefix contains any of the forbidden characters
    assert!(vanity_prefix.contains("l") == false); // lowercase L
    assert!(vanity_prefix.contains("I") == false); // uppercase i
    assert!(vanity_prefix.contains("0") == false); // zero
    assert!(vanity_prefix.contains("O") == false); // uppercase o

    // convert the vanity prefix to a byte array
    let vanity_prefix_bytes = vanity_prefix.as_bytes();
    let vanity_prefix_len: usize = vanity_prefix_bytes.len();
    
    let ctx = CudaContext::new(ordinal)?;
    ctx.bind_to_thread()?;
    let stream = ctx.default_stream();

    // Load the pre-compiled PTX that was generated during build
    println!("[{ordinal}] Loading module...");
    let module = ctx.load_module(Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"))))?;
    let f = module.load_function("find_vanity_private_key").unwrap();

    // Configure kernel launch parameters
    let num_blocks = 1024;  // Number of blocks    
    let block_size = 128; // Threads per block
    let iterations_per_thread: usize = 1; // Number of iterations per thread
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut attempts = 0;
    println!("[{ordinal}] Starting search loop...");

    let mut rng = rand::thread_rng();
    let start_time = Instant::now();
    let mut matches_found = 0;

    loop {
        attempts += 1;
        let rng_seed: u64 = rng.r#gen::<u64>();
        
        let mut found_flag = [0.0f32; 1];
        let mut found_private_key = [0u8; 32];
        let mut found_public_key = [0u8; 32];
        let mut found_bs58_encoded_public_key = [0u8; 44];
        
        let vanity_prefix_dev = stream.memcpy_stod(vanity_prefix_bytes)?;
        let found_flag_dev = stream.memcpy_stod(&found_flag)?;
        let found_private_key_dev = stream.memcpy_stod(&found_private_key)?;
        let found_public_key_dev = stream.memcpy_stod(&found_public_key)?;
        let found_bs58_encoded_public_key_dev = stream.memcpy_stod(&found_bs58_encoded_public_key)?;    
        
        // Launch the kernel
        let mut launch_args = stream.launch_builder(&f);
        launch_args.arg(&vanity_prefix_dev);
        launch_args.arg(&vanity_prefix_len);
        launch_args.arg(&rng_seed);
        launch_args.arg(&iterations_per_thread);
        launch_args.arg(&found_flag_dev);
        launch_args.arg(&found_private_key_dev);
        launch_args.arg(&found_public_key_dev);
        launch_args.arg(&found_bs58_encoded_public_key_dev);

        unsafe { launch_args.launch(cfg) }?;

        stream.synchronize()?;

        // Check if we found a match
        stream.memcpy_dtoh(&found_flag_dev, &mut found_flag)?;
        
        if found_flag[0] != 0.0 {
            // We found a match! Copy results back to host
            stream.memcpy_dtoh(&found_private_key_dev, &mut found_private_key)?;
            stream.memcpy_dtoh(&found_public_key_dev, &mut found_public_key)?;
            stream.memcpy_dtoh(&found_bs58_encoded_public_key_dev, &mut found_bs58_encoded_public_key)?;

            println!("[{ordinal}] Found private key: {:02x?}", found_private_key);
            println!("[{ordinal}] Found public key: {:02x?}", found_public_key);
            println!("[{ordinal}] Found bs58 encoded public key: {:02x?}", found_bs58_encoded_public_key);

            matches_found += 1;
            let elapsed = start_time.elapsed();
            let matches_per_second = matches_found as f64 / elapsed.as_secs_f64();
            println!("[{ordinal}] Found {matches_found} matches in {elapsed:?} ({matches_per_second:.2} matches/sec) with {attempts} attempts num_blocks = {num_blocks} block_size = {block_size} iterations_per_thread = {iterations_per_thread}");
        }
    }
}

fn main() -> Result<(), DriverError> {
    // Define the vanity prefix we're looking for
    let args = std::env::args().collect::<Vec<String>>();
    let vanity_prefix = args[1].to_string();

    // Initialize CUDA context and get default stream
    let num_devices = CudaContext::device_count()?;
    println!("Found {} CUDA devices", num_devices);

    let mut handles = Vec::new();
    for i in 0..num_devices as usize {
        println!("Starting device {}", i);
        let vanity_prefix_clone = vanity_prefix.clone();
        let handle = std::thread::spawn(move || device_main(i, vanity_prefix_clone));
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap()?;
    }

    Ok(())
}
