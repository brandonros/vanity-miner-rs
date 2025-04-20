use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use rand::Rng;
use std::time::Instant;

fn main() -> Result<(), DriverError> {
    // Define the vanity prefix we're looking for
    let vanity_prefix = "aa";

    // check if the vanity prefix contains any of the forbidden characters
    assert!(vanity_prefix.contains("l") == false); // lowercase L
    assert!(vanity_prefix.contains("I") == false); // uppercase i
    assert!(vanity_prefix.contains("0") == false); // zero
    assert!(vanity_prefix.contains("O") == false); // uppercase o

    // convert the vanity prefix to a byte array
    let vanity_prefix_bytes = vanity_prefix.as_bytes();
    let vanity_prefix_len: usize = vanity_prefix_bytes.len();

    // Initialize CUDA context and get default stream
    let ctx = CudaContext::new(0)?;
    ctx.bind_to_thread()?;
    let stream = ctx.default_stream();

    // Load the pre-compiled PTX that was generated during build
    println!("Loading module...");
    let module = ctx.load_module(Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"))))?;
    let f = module.load_function("find_vanity_private_key").unwrap();

    // Configure kernel launch parameters
    let block_size = 256; // Threads per block
    let num_blocks = 64;  // Number of blocks
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut attempts = 0;
    println!("Starting search loop...");

    let mut rng = rand::thread_rng();
    let max_num_iterations: usize = 10000;
    let total_hashes_per_attempt = block_size as u64 * num_blocks as u64 * max_num_iterations as u64;
    
    loop {
        attempts += 1;
        let rng_seed: u64 = rng.r#gen::<u64>();
        let start_time = Instant::now();

        let mut found_flag = [0.0f32; 1];
        let mut found_private_key = [0u8; 32];
        let mut found_public_key = [0u8; 32];
        let mut found_bs58_encoded_public_key = [0u8; 44];
        
        let vanity_prefix_dev    = stream.memcpy_stod(vanity_prefix_bytes)?;
        let found_flag_dev = stream.memcpy_stod(&found_flag)?;
        let found_private_key_dev = stream.memcpy_stod(&found_private_key)?;
        let found_public_key_dev = stream.memcpy_stod(&found_public_key)?;
        let found_bs58_encoded_public_key_dev = stream.memcpy_stod(&found_bs58_encoded_public_key)?;    
        
        // Launch the kernel
        let mut launch_args = stream.launch_builder(&f);
        launch_args.arg(&vanity_prefix_dev);
        launch_args.arg(&vanity_prefix_len);
        launch_args.arg(&rng_seed);
        launch_args.arg(&max_num_iterations);
        launch_args.arg(&found_flag_dev);
        launch_args.arg(&found_private_key_dev);
        launch_args.arg(&found_public_key_dev);
        launch_args.arg(&found_bs58_encoded_public_key_dev);

        println!("Attempt {}: Launching kernel with seed {}...", attempts, rng_seed);
        unsafe { launch_args.launch(cfg) }?;

        stream.synchronize()?;

        let elapsed = start_time.elapsed();
        let hash_rate = total_hashes_per_attempt as f64 / elapsed.as_secs_f64();
        println!("Attempt {} took {:.2?} ({:.2}M hashes/sec)", 
                attempts, 
                elapsed,
                hash_rate / 1_000_000.0);

        // Check if we found a match
        stream.memcpy_dtoh(&found_flag_dev, &mut found_flag)?;
        
        if found_flag[0] != 0.0 {
            // We found a match! Copy results back to host
            stream.memcpy_dtoh(&found_private_key_dev, &mut found_private_key)?;
            stream.memcpy_dtoh(&found_public_key_dev, &mut found_public_key)?;
            stream.memcpy_dtoh(&found_bs58_encoded_public_key_dev, &mut found_bs58_encoded_public_key)?;

            println!("\nFound match after {} attempts!", attempts);
            println!("Found private key: {:02x?}", found_private_key);
            println!("Found public key: {:02x?}", found_public_key);
            println!("Found bs58 encoded public key: {:02x?}", found_bs58_encoded_public_key);
            break;
        }
    }

    Ok(())
}
