use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use rand::Rng;
use std::time::Instant;

fn device_main(ordinal: usize, vanity_prefix: String, blocks_per_grid: usize, threads_per_block: usize) -> Result<(), DriverError> {
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
    let cfg = LaunchConfig {
        grid_dim: (blocks_per_grid as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: 0,
    };

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

            // print
            let wallet_formatted_result = hex::encode([found_private_key, found_public_key].concat());
            let public_key_string = String::from_utf8(found_bs58_encoded_public_key).unwrap();
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

fn main() -> Result<(), DriverError> {
    // Define the vanity prefix we're looking for
    let args = std::env::args().collect::<Vec<String>>();
    let vanity_prefix = args[1].to_string();
    let blocks_per_grid = args[2].parse::<usize>().unwrap();
    let threads_per_block = args[3].parse::<usize>().unwrap();

    // Initialize CUDA context and get default stream
    let num_devices = CudaContext::device_count()?;
    println!("Found {} CUDA devices", num_devices);

    let mut handles = Vec::new();
    for i in 0..num_devices as usize {
        println!("Starting device {}", i);
        let vanity_prefix_clone = vanity_prefix.clone();
        let handle = std::thread::spawn(move || device_main(i, vanity_prefix_clone, blocks_per_grid, threads_per_block));
        handles.push(handle);
    }
    for handle in handles {
        handle.join().unwrap()?;
    }

    Ok(())
}
