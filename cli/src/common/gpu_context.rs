use cuda_core::sys::{
    CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    CUlimit_enum_CU_LIMIT_STACK_SIZE, cuCtxSetLimit, cuDeviceGetAttribute,
};
use cuda_core::{CudaContext, CudaStream, IntoResult};
use std::error::Error;
use std::ffi::c_int;
use std::mem::MaybeUninit;
use std::sync::Arc;

pub struct GpuContext {
    pub stream: Arc<CudaStream>,
    pub blocks_per_grid: usize,
    pub threads_per_block: usize,
    pub operations_per_launch: usize,
}

impl GpuContext {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, Box<dyn Error + Send + Sync>> {
        ctx.bind_to_thread()?;

        if let Ok(stack_size) = std::env::var("STACK_SIZE") {
            let stack_size = stack_size.parse::<usize>()?;
            unsafe {
                cuCtxSetLimit(CUlimit_enum_CU_LIMIT_STACK_SIZE, stack_size).result()?;
            }
        }

        let stream = ctx.new_stream()?;

        let number_of_streaming_multiprocessors = unsafe {
            let mut count = MaybeUninit::<c_int>::uninit();
            cuDeviceGetAttribute(
                count.as_mut_ptr(),
                CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                ctx.cu_device(),
            )
            .result()?;
            count.assume_init() as usize
        };

        let blocks_per_sm = std::env::var("BLOCKS_PER_SM")
            .unwrap_or_else(|_| "128".to_string())
            .parse::<usize>()?;
        let threads_per_block = std::env::var("THREADS_PER_BLOCK")
            .unwrap_or_else(|_| "256".to_string())
            .parse::<usize>()?;
        let blocks_per_grid = number_of_streaming_multiprocessors * blocks_per_sm;
        let operations_per_launch = blocks_per_grid * threads_per_block;

        Ok(Self {
            stream,
            blocks_per_grid,
            threads_per_block,
            operations_per_launch,
        })
    }

    pub fn print_launch_info(&self, ordinal: usize, mode_name: &str) {
        println!(
            "[{ordinal}] Starting {mode_name} search loop ({} blocks per grid, {} threads per block, {} operations per launch)",
            self.blocks_per_grid, self.threads_per_block, self.operations_per_launch
        );
    }
}
