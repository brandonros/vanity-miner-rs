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
        let mut results = [0u32; logic::SELF_TEST_NUM_CHECKS];

        // Launch each slot's kernel sequentially with a copy+sync between
        // launches. If a kernel hits an illegal address, the *next* host
        // call (kernel launch or copy) is what surfaces the sticky error —
        // so per-slot sync lets us pinpoint which slot's kernel faulted.
        // Slots before the fault still produce reliable values; slots at
        // and after are reported as FAIL since their writes never landed.
        macro_rules! run_slot {
            ($slot:expr, $method:ident) => {{
                let slot: usize = $slot;
                let label = logic::SELF_TEST_LABELS[slot];
                let launch_res = unsafe {
                    module.$method(stream, launch_config, &mut results_dev)
                };
                if let Err(e) = launch_res {
                    return Err(format!(
                        "[{ordinal}] slot {slot:2} [{label}] launch failed: {e}"
                    )
                    .into());
                }
                if let Err(e) = results_dev.copy_to_host(stream, &mut results) {
                    return Err(format!(
                        "[{ordinal}] slot {slot:2} [{label}] DtoH copy failed (kernel likely faulted): {e}"
                    )
                    .into());
                }
                if let Err(e) = stream.synchronize() {
                    return Err(format!(
                        "[{ordinal}] slot {slot:2} [{label}] sync failed (kernel likely faulted): {e}"
                    )
                    .into());
                }
                let status = if results[slot] == 1 { "PASS" } else { "FAIL" };
                println!("[{ordinal}] slot {slot:2} {status}  {label}");
            }};
        }

        // Slots 0-3: solana per-primitive bisect (run first so a primitive
        // fault localizes before the composed solana kernels inline it).
        run_slot!(0,  kernel_self_test_primitive_xoroshiro);
        run_slot!(1,  kernel_self_test_primitive_sha512);
        run_slot!(2,  kernel_self_test_primitive_ed25519);
        run_slot!(3,  kernel_self_test_primitive_base58);
        // Slots 4-9: non-solana primitive bisect — same idea as 0-3 but for
        // the primitives consumed by the bitcoin / ethereum / shallenge /
        // WIF pipelines. Run before the composed kernels so a fault
        // localizes to the broken primitive instead of the inlined caller.
        run_slot!(4,  kernel_self_test_primitive_secp256k1_compressed);
        run_slot!(5,  kernel_self_test_primitive_secp256k1_uncompressed);
        run_slot!(6,  kernel_self_test_primitive_keccak256);
        run_slot!(7,  kernel_self_test_primitive_ripemd160);
        run_slot!(8,  kernel_self_test_primitive_sha256_32);
        run_slot!(9,  kernel_self_test_primitive_sha256_variable);
        // Slots 10-30: composed-subsystem KAT checks.
        run_slot!(10, kernel_self_test_solana_priv);
        run_slot!(11, kernel_self_test_solana_pub);
        run_slot!(12, kernel_self_test_solana_encoded);
        run_slot!(13, kernel_self_test_ethereum_priv);
        run_slot!(14, kernel_self_test_ethereum_pub);
        run_slot!(15, kernel_self_test_ethereum_address);
        run_slot!(16, kernel_self_test_bitcoin_priv);
        run_slot!(17, kernel_self_test_bitcoin_pub);
        run_slot!(18, kernel_self_test_bitcoin_pkh);
        run_slot!(19, kernel_self_test_bitcoin_encoded);
        run_slot!(20, kernel_self_test_bitcoin_matches);
        run_slot!(21, kernel_self_test_wif_compressed_mainnet);
        run_slot!(22, kernel_self_test_wif_uncompressed_mainnet);
        run_slot!(23, kernel_self_test_wif_compressed_testnet);
        run_slot!(24, kernel_self_test_wif_uncompressed_testnet);
        run_slot!(25, kernel_self_test_shallenge_hash);
        run_slot!(26, kernel_self_test_shallenge_nonce_len);
        run_slot!(27, kernel_self_test_shallenge_is_better);
        run_slot!(28, kernel_self_test_compare_hashes_lt);
        run_slot!(29, kernel_self_test_compare_hashes_gt);
        run_slot!(30, kernel_self_test_compare_hashes_eq);

        report(&results)
    }
}
