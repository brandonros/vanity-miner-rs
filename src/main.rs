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
    
    // Copy vanity prefix to device
    let vanity_prefix_dev    = stream.memcpy_stod(vanity_prefix)?;
    let found_flag_dev       = stream.memcpy_stod(&found_flag)?;
    // Launch the kernel
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&vanity_prefix_dev);
    launch_args.arg(&vanity_prefix_len);
    launch_args.arg(&rng_seed);
    launch_args.arg(&found_flag_dev);

    println!("Launching kernel...");
    unsafe { launch_args.launch(cfg) }?;

    println!("Synchronizing...");
    stream.synchronize()?;

    Ok(())
}
