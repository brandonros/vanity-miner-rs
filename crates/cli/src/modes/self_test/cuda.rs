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
    let cases = args.selected()?;
    let mut cache = self_test::DeviceResults::default();
    self_test::run(&format!("CUDA {ordinal}"), &cases, |case| {
        cache.check(case, || {
            let stream = &gpu.stream;
            let mut results = [self_test::SENTINEL; logic::self_test::SELF_TEST_NUM_CHECKS];
            let device = DeviceBuffer::from_slice(&results).map_err(|e| e.to_string())?;
            let module = crate::runner::cuda::module::load_self_test_module(ordinal, case.kernel)
                .map_err(|e| e.to_string())?;
            let kernel = module
                .get_function(case.kernel)
                .map_err(|e| e.to_string())?;
            unsafe { launch!(kernel<<<1u32, 1u32, 0, stream>>>(device.as_device_ptr())) }
                .map_err(|e| e.to_string())?;
            stream.synchronize().map_err(|e| e.to_string())?;
            device.copy_to(&mut results).map_err(|e| e.to_string())?;
            Ok(results.to_vec())
        })
    })
    .map_err(Into::into)
}
