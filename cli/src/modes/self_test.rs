use std::error::Error;

fn report(results: &[u32; logic::SELF_TEST_NUM_CHECKS]) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut failed = 0usize;
    for (i, &r) in results.iter().enumerate() {
        let status = if r == 1 { "PASS" } else { "FAIL" };
        if r != 1 {
            failed += 1;
        }
        println!("  [{:2}] {}  {}", i, status, logic::SELF_TEST_LABELS[i]);
    }
    if failed == 0 {
        println!("self-test: all {} checks passed", results.len());
        Ok(())
    } else {
        Err(format!("self-test: {} / {} checks failed", failed, results.len()).into())
    }
}

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;

    pub fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Running self-test on CPU");
        let mut results = [0u32; logic::SELF_TEST_NUM_CHECKS];
        logic::run_self_test(&mut results);
        report(&results)
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use crate::common::GpuContext;
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
    use kernels::kernels::LoadedModule;
    use std::sync::Arc;

    pub fn run(
        ordinal: usize,
        ctx: &Arc<CudaContext>,
        module: &LoadedModule,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let _ = ctx;
        let gpu = GpuContext::new(ctx)?;
        println!("[{}] Running self-test on GPU", ordinal);

        let stream = &gpu.stream;
        let mut results_dev =
            DeviceBuffer::<u32>::zeroed(stream, logic::SELF_TEST_NUM_CHECKS)?;

        let launch_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            module.kernel_self_test(stream, launch_config, &mut results_dev)?;
        }

        let mut results = [0u32; logic::SELF_TEST_NUM_CHECKS];
        results_dev.copy_to_host(stream, &mut results)?;
        stream.synchronize()?;

        report(&results)
    }
}
