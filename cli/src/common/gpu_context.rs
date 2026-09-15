use cust::context::ResourceLimit;
use cust::device::Device;
use cust::prelude::Context;
use cust::stream::{Stream, StreamFlags};
use std::error::Error;

pub struct GpuContext {
    pub module: cust::module::Module,
    pub stream: Stream,
    pub blocks_per_grid: usize,
    pub threads_per_block: usize,
    pub operations_per_launch: usize,
    // Keep context alive for the lifetime of GpuContext
    #[allow(dead_code)]
    ctx: Context,
}

impl GpuContext {
    pub fn configured_threads_per_block() -> Result<usize, Box<dyn Error + Send + Sync>> {
        let threads = std::env::var("THREADS_PER_BLOCK")
            .unwrap_or_else(|_| "256".to_string())
            .parse::<usize>()?;
        if !(1..=1024).contains(&threads) {
            return Err("THREADS_PER_BLOCK must be between 1 and 1024".into());
        }
        Ok(threads)
    }

    pub fn new(ordinal: usize, module: &str) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Self::with_module(ordinal, || super::cuda_module::load_module(ordinal, module))
    }

    #[cfg(feature = "self_test_support")]
    pub fn for_self_test(ordinal: usize) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Self::with_module(ordinal, || {
            super::cuda_module::load_self_test_module(
                ordinal,
                vanity_miner::self_test_suite::inventory()[0].kernel,
            )
        })
    }

    fn with_module(
        ordinal: usize,
        load: impl FnOnce() -> Result<cust::module::Module, Box<dyn Error + Send + Sync>>,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let device = Device::get_device(ordinal as u32)?;
        let ctx = Context::new(device)?;
        cust::context::CurrentContext::set_current(&ctx)?;

        // Optionally override stack size
        if let Ok(stack_size) = std::env::var("STACK_SIZE") {
            let stack_size = stack_size.parse::<usize>()?;
            cust::context::CurrentContext::set_resource_limit(
                ResourceLimit::StackSize,
                stack_size,
            )?;
        } else {
            // CUDA's default per-thread stack is 1024 bytes. Rust-CUDA's NVVM
            // backend aggressively inlines whole pipelines, so any kernel that
            // composes k256/dalek + xoroshiro needs much more — bisected at
            // 8 KiB FAIL / 16 KiB PASS for the eth-priv-bisect's simplest
            // composed kernel. The full self-test ladder has bigger kernels
            // (depot up to 1856 bytes + deeper k256/dalek call chains), so
            // give 2× headroom over the measured floor.
            cust::context::CurrentContext::set_resource_limit(
                ResourceLimit::StackSize,
                if cfg!(feature = "crypto-cli") || cfg!(feature = "self_test_support") {
                    65536
                } else {
                    16384
                },
            )?;
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let number_of_streaming_multiprocessors =
            device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as usize;
        let blocks_per_sm = std::env::var("BLOCKS_PER_SM")
            .unwrap_or_else(|_| "128".to_string())
            .parse::<usize>()?;
        let threads_per_block = Self::configured_threads_per_block()?;
        let blocks_per_grid = number_of_streaming_multiprocessors * blocks_per_sm;
        let operations_per_launch = blocks_per_grid * threads_per_block;

        let module = load()?;
        Ok(Self {
            module,
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

impl Drop for GpuContext {
    fn drop(&mut self) {
        let _ = cust::context::CurrentContext::set_current(&self.ctx);
        let _ = self.stream.synchronize();
    }
}
