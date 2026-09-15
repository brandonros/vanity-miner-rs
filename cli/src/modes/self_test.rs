use std::error::Error;
use vanity_miner::self_test_suite::{self, Outcome};

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;
    pub fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut results = [0; logic::self_test::SELF_TEST_NUM_CHECKS];
        logic::self_test::run_self_test(&mut results);
        self_test_suite::run("CPU", |case| {
            let slot = case.slot;
            if results[slot] != 1 {
                return Err("known-answer mismatch".into());
            }
            Ok(Outcome::Passed)
        })
        .map_err(Into::into)
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use crate::common::GpuContext;
    use cust::{
        launch,
        memory::{CopyDestination, DeviceBuffer},
    };
    pub fn run(ordinal: usize, gpu: &GpuContext) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut cache = self_test_suite::DeviceResults::default();
        self_test_suite::run(&format!("CUDA {ordinal}"), |case| {
            cache.check(case, || {
                let stream = &gpu.stream;
                let mut results =
                    [self_test_suite::SENTINEL; logic::self_test::SELF_TEST_NUM_CHECKS];
                let device = DeviceBuffer::from_slice(&results).map_err(|e| e.to_string())?;
                let module =
                    crate::common::cuda_module::load_self_test_module(ordinal, case.kernel)
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
}
