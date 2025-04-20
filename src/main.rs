use cust::prelude::*;
use nanorand::{Rng, WyRand};
use std::error::Error;

/// How many numbers to generate and add together.
const NUMBERS_LEN: usize = 100_000;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    // generate our random vectors.
    let mut wyrand = WyRand::new();
    let mut lhs = vec![0u8; 32];
    wyrand.fill(&mut lhs);

    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    println!("initializing CUDA");
    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    println!("making CUDA module");
    let module = Module::from_ptx(PTX, &[])?;

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    println!("making CUDA stream");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // allocate the GPU memory needed to house our numbers and copy them over.
    println!("allocating GPU memory");
    let lhs_gpu = lhs.as_slice().as_dbuf()?;

    // retrieve the `find_private_key` kernel from the module so we can calculate the right launch config.
    println!("retrieving CUDA kernel");
    let find_private_key = module.get_function("find_private_key")?;

    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    println!("calculating launch configuration");
    let (_, block_size) = find_private_key.suggested_launch_configuration(0, 0.into())?;
    let grid_size = (NUMBERS_LEN as u32).div_ceil(block_size);
    println!(
        "using {} blocks and {} threads per block",
        grid_size, block_size
    );

    // TODO: remove later, just for testing
    let grid_size = 1;
    let block_size = 1;

    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        println!("launching CUDA kernel");
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            find_private_key<<<grid_size, block_size, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
            )
        )?;
        println!("kernel launched");
    }

    println!("synchronizing stream");
    stream.synchronize()?;
    println!("stream synchronized");

    Ok(())
}
