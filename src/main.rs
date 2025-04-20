use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

fn main() -> Result<(), DriverError> {
    // Define the vanity prefix we're looking for
    let vanity_prefix = b"aaaa";
    let vanity_prefix_len = vanity_prefix.len();
    let rng_seed: u64 = 2457272140905572020; // You might want to generate this randomly

    // Initialize CUDA context and get default stream
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Load the pre-compiled PTX that was generated during build
    let module = ctx.load_module(Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"))))?;
    let f = module.load_function("find_vanity_private_key").unwrap();

    // Configure kernel launch parameters
    let n = 1; // Number of parallel threads to run
    let cfg = LaunchConfig::for_num_elems(n as u32);
    
    // Copy vanity prefix to device
    let vanity_prefix_dev = stream.memcpy_stod(vanity_prefix)?;

    // Launch the kernel
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&vanity_prefix_dev);
    launch_args.arg(&vanity_prefix_len);
    launch_args.arg(&rng_seed);
    unsafe { launch_args.launch(cfg) }?;

    stream.synchronize()?;

    Ok(())
}
