use cust::context::ResourceLimit;
use cust::device::Device;
use cust::prelude::Context;
use cust::stream::{Stream, StreamFlags};
use std::error::Error;

pub struct GpuContext {
    pub stream: Stream,
    pub blocks_per_grid: usize,
    pub threads_per_block: usize,
    pub operations_per_launch: usize,
    // Keep context alive for the lifetime of GpuContext
    #[allow(dead_code)]
    ctx: Context,
}

impl GpuContext {
    pub fn new(ordinal: usize) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let device = Device::get_device(ordinal as u32)?;
        let ctx = Context::new(device)?;
        cust::context::CurrentContext::set_current(&ctx)?;

        // Optionally override stack size
        if let Some(stack_size) = std::env::var("STACK_SIZE").ok() {
            let stack_size = stack_size.parse::<usize>().unwrap();
            cust::context::CurrentContext::set_resource_limit(ResourceLimit::StackSize, stack_size)?;
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let number_of_streaming_multiprocessors =
            device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as usize;
        let blocks_per_sm = std::env::var("BLOCKS_PER_SM")
            .unwrap_or("128".to_string())
            .parse::<usize>()
            .unwrap();
        let threads_per_block = std::env::var("THREADS_PER_BLOCK")
            .unwrap_or("256".to_string())
            .parse::<usize>()
            .unwrap();
        let blocks_per_grid = number_of_streaming_multiprocessors * blocks_per_sm;
        let operations_per_launch = blocks_per_grid * threads_per_block;

        Ok(Self {
            stream,
            blocks_per_grid,
            threads_per_block,
            operations_per_launch,
            ctx,
        })
    }

    pub fn print_launch_info(&self, ordinal: usize, mode_name: &str) {
        println!(
            "[{ordinal}] Starting {mode_name} search loop ({} blocks per grid, {} threads per block, {} operations per launch)",
            self.blocks_per_grid, self.threads_per_block, self.operations_per_launch
        );
    }
}
