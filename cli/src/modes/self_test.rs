use std::error::Error;
use vanity_miner::self_test_suite::{self, Outcome};

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;
    pub fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut results = [0; logic::SELF_TEST_NUM_CHECKS];
        logic::run_self_test(&mut results);
        self_test_suite::run("CPU", |case| {
            let Some(slot) = case.slot else {
                return Ok(Outcome::Skipped("requires GPU launch"));
            };
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
        self_test_suite::run(&format!("CUDA {ordinal}"), |case| {
            let stream = &gpu.stream;
            let mut results = [0xa5a5a5a5u32; logic::SELF_TEST_NUM_CHECKS];
            let device = DeviceBuffer::from_slice(&results).map_err(|e| e.to_string())?;
            let kernel = gpu
                .module
                .get_function(case.kernel)
                .map_err(|e| e.to_string())?;
            unsafe { launch!(kernel<<<1u32, 1u32, 0, stream>>>(device.as_device_ptr())) }
                .map_err(|e| e.to_string())?;
            stream.synchronize().map_err(|e| e.to_string())?;
            device.copy_to(&mut results).map_err(|e| e.to_string())?;
            let slot = case.slot.unwrap_or(0);
            for (index, value) in results.into_iter().enumerate() {
                let expected = if index == slot { 1 } else { 0xa5a5a5a5 };
                if value != expected {
                    return Err(format!("result slot {index} mismatch"));
                }
            }
            Ok(Outcome::Passed)
        })
        .map_err(Into::into)
    }

}
