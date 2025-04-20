use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    // Define the vanity prefix we're looking for
    let vanity_prefix    = b"aa";
    let vanity_prefix_len: usize = vanity_prefix.len();
    let rng_seed: u64 = 2457272140905572020; // You might want to generate this randomly
    let mut found_flag = [0u8; 1];
    let mut found_private_key = [0u8; 32];
    let mut found_public_key = [0u8; 32];
    let mut found_bs58_encoded_public_key = [0u8; 44];

    // Initialize CUDA context and get default stream
    let ctx = CudaContext::new(0)?;
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

    println!("Copying to device...");
    
    // Copy slices to device
    let vanity_prefix_dev    = stream.memcpy_stod(vanity_prefix)?;
    let found_flag_dev       = stream.memcpy_stod(&found_flag)?;
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

    println!("Launching kernel...");
    unsafe { launch_args.launch(cfg) }?;

    println!("Synchronizing...");
    stream.synchronize()?;

    // Copy results back to host
    stream.memcpy_dtoh(&found_flag_dev, &mut found_flag)?;
    stream.memcpy_dtoh(&found_private_key_dev, &mut found_private_key)?;
    stream.memcpy_dtoh(&found_public_key_dev, &mut found_public_key)?;
    stream.memcpy_dtoh(&found_bs58_encoded_public_key_dev, &mut found_bs58_encoded_public_key)?;

    println!("Found flag: {:?}", found_flag);
    println!("Found private key: {:02x?}", found_private_key);
    println!("Found public key: {:02x?}", found_public_key);
    println!("Found bs58 encoded public key: {:02x?}", found_bs58_encoded_public_key);

    Ok(())
}
