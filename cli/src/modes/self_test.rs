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

        let launch_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        // Plumbing probe: confirm PTX load + launch + DtoH copy work before
        // running the heavy logic body. If this fails, the bug isn't in
        // `logic::run_self_test` — it's in the kernel-load / launch path.
        let mut stub_dev = DeviceBuffer::<u32>::zeroed(stream, 1)?;
        unsafe {
            module.kernel_self_test_stub(stream, launch_config, &mut stub_dev)?;
        }
        let mut stub = [0u32; 1];
        stub_dev.copy_to_host(stream, &mut stub)?;
        stream.synchronize()?;
        if stub[0] != 1 {
            return Err(format!(
                "[{ordinal}] stub kernel wrote {} (expected 1) — PTX/launch plumbing is broken",
                stub[0]
            )
            .into());
        }
        println!("[{ordinal}] stub kernel OK (results[0] = 1)");

        let mut results_dev =
            DeviceBuffer::<u32>::zeroed(stream, logic::SELF_TEST_NUM_CHECKS)?;

        unsafe {
            module.kernel_self_test(stream, launch_config, &mut results_dev)?;
        }

        let mut results = [0u32; logic::SELF_TEST_NUM_CHECKS];
        results_dev.copy_to_host(stream, &mut results)?;
        stream.synchronize()?;

        report(&results)
    }
}
