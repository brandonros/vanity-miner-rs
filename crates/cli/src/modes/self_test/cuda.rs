use super as self_test;
use crate::runner::cuda::context::GpuContext;
use cust::{
    launch,
    memory::{CopyDestination, DeviceBuffer},
};
use std::error::Error;
pub fn run(
    ordinal: usize,
    gpu: &GpuContext,
    args: &super::args::SelfTestArgs,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    self_test::run(&format!("CUDA {ordinal}"), &args.selected()?, |mode| {
        let stream = &gpu.stream;
        let mut results = vec![self_test::SENTINEL; mode.checks.len()];
        let device = DeviceBuffer::from_slice(&results).map_err(|e| e.to_string())?;
        let module = crate::runner::cuda::module::load_self_test_module(ordinal, mode.name)
            .map_err(|e| e.to_string())?;
        let kernel = module
            .get_function(&format!("kernel_self_test_{}", mode.name))
            .map_err(|e| e.to_string())?;
        unsafe { launch!(kernel<<<1u32, 1u32, 0, stream>>>(device.as_device_ptr())) }
            .map_err(|e| e.to_string())?;
        stream.synchronize().map_err(|e| e.to_string())?;
        device.copy_to(&mut results).map_err(|e| e.to_string())?;
        Ok(results)
    })
    .map_err(Into::into)
}
