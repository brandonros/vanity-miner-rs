use cust::device::Device;
use cust::module::Module;
use cust::prelude::Context;
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt;
use cust::memory::CopyDestination;
use cust::launch;
use std::error::Error;

pub fn device_main_add(
    ordinal: usize, 
    module: &Module,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Initialize device and context
    let device = Device::get_device(ordinal as u32)?;
    let ctx = Context::new(device)?;
    cust::context::CurrentContext::set_current(&ctx)?;
    
    // Create stream and get kernel function
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel_add = module.get_function("kernel_add")?;

    // Calculate grid/block dimensions
    let number_of_streaming_multiprocessors = device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as usize;
    let blocks_per_sm = std::env::var("BLOCKS_PER_SM").unwrap_or("128".to_string()).parse::<usize>().unwrap();
    let threads_per_block = std::env::var("THREADS_PER_BLOCK").unwrap_or("256".to_string()).parse::<usize>().unwrap();
    let blocks_per_grid = number_of_streaming_multiprocessors * blocks_per_sm;
    let operations_per_launch = blocks_per_grid * threads_per_block;

    let data_len = operations_per_launch;
    
    // Initialize input arrays with predictable data
    let input_a: Vec<f32> = (0..data_len).map(|i| (i % 1000) as f32).collect();
    let input_b: Vec<f32> = (0..data_len).map(|i| ((i + 1) % 1000) as f32).collect();

    println!("[{ordinal}] Processing {} elements ({} blocks, {} threads per block)", 
                data_len, blocks_per_grid, threads_per_block);

    // Prepare output buffer
    let mut output = vec![0.0f32; data_len];
    
    // Transfer data to GPU using SliceExt
    let input_a_dev = input_a.as_slice().as_dbuf()?;
    let input_b_dev = input_b.as_slice().as_dbuf()?;
    let output_dev = output.as_slice().as_dbuf()?;

    // Launch kernel
    unsafe {
        launch!(
            kernel_add<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                input_a_dev.as_device_ptr(),
                input_b_dev.as_device_ptr(),
                output_dev.as_device_ptr(),
            )
        )?;
    }

    // Wait for completion and copy result back
    stream.synchronize()?;
    output_dev.copy_to(&mut output)?;

    println!("[{ordinal}] Computation completed");

    // print the first 10 elements of the output
    println!("[{ordinal}] First 10 elements of the output: {:?}", &output[..10]);
    
    Ok(())
}
