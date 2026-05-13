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
        // Slots 31-40: raw-arithmetic micro-bisect (one PTX op per slot).
        run_slot!(31, kernel_self_test_arith_u32_div_var);
        run_slot!(32, kernel_self_test_arith_u32_div_const);
        run_slot!(33, kernel_self_test_arith_u64_div_var);
        run_slot!(34, kernel_self_test_arith_u64_div_const);
        run_slot!(35, kernel_self_test_arith_u32_rem_var);
        run_slot!(36, kernel_self_test_arith_u64_rem_var);
        run_slot!(37, kernel_self_test_arith_u32_mul_lo);
        run_slot!(38, kernel_self_test_arith_u64_mul_lo);
        run_slot!(39, kernel_self_test_arith_u64_mul_hi);
        run_slot!(40, kernel_self_test_arith_u128_mul);
        // Slot 44: xoroshiro base64 nonce (uses `&'static [u8]` slice
        // for the alphabet — FAILs with wrong bytes but doesn't crash).
        // Slot 45 (bech32) and slot 43 (base58 all-zeros) turn out to be
        // crash-prone too — both use `&'static [u8; N]` array references
        // for their alphabets, same shape as slot 41. Deferred to the
        // tail; see the comment near their run_slot! calls.
        run_slot!(44, kernel_self_test_xoroshiro_base64_nonce);
        // Slots 46-56: tier-2 arithmetic bisect targeting PTX idioms
        // dalek/k256 hit that the tier-1 net (31-40) doesn't cover.
        run_slot!(46, kernel_self_test_arith_overflowing_add);
        run_slot!(47, kernel_self_test_arith_overflowing_sub);
        run_slot!(48, kernel_self_test_arith_carry_chain_3limb);
        run_slot!(49, kernel_self_test_arith_widening_mul_pair);
        run_slot!(50, kernel_self_test_arith_mad_lo_u64);
        run_slot!(51, kernel_self_test_arith_mad_hi_u64);
        run_slot!(52, kernel_self_test_arith_mul_wide_u32);
        run_slot!(53, kernel_self_test_arith_mask_blend_true);
        run_slot!(54, kernel_self_test_arith_mask_blend_false);
        run_slot!(55, kernel_self_test_arith_var_shr_u64);
        run_slot!(56, kernel_self_test_arith_var_shl_u64);
        // Slots 57-58: black_box identity probes. Cheap one-line tests
        // that determine whether `core::hint::black_box` preserves its
        // input — if not, every tier-1/tier-2 arith slot's FAIL is a
        // black_box artifact, not a per-op codegen bug.
        run_slot!(57, kernel_self_test_arith_blackbox_identity_u64);
        run_slot!(58, kernel_self_test_arith_blackbox_identity_u32);
        // Slot 59: isolated divmod-by-58 — must run BEFORE the deferred
        // base58 kernels so its result is captured even when they fault.
        run_slot!(59, kernel_self_test_base58_div_by_58);
        // Slots 60-62: triangulating bisects for the iter_mut + alphabet-
        // lookup pattern (the suspect shared crash path between slot 41
        // and slot 43). Ordered simplest→most complex so an early crash
        // still captures earlier slots' results.
        // Slot 63 (slice counterpart) runs BEFORE slot 60 (array-ref
        // version) so we capture the slice baseline even if 60 faults
        // the way bech32/base58 did. If 63 PASSes and 60 CRASHes, the
        // array-ref-vs-slice discriminator is confirmed.
        run_slot!(63, kernel_self_test_iter_static_slice_lookup);
        run_slot!(61, kernel_self_test_iter_mut_slice_partial);
        run_slot!(60, kernel_self_test_iter_static_table_lookup);
        run_slot!(62, kernel_self_test_iter_mut_alphabet_lookup);
        // Crash-tail slots: 41, 42, 43 (base58 — `&[u8; 58]` alphabet),
        // 45 (bech32 — `&[u8; 32]` alphabet). All four faulted with the
        // same `0xfff…` wild-pointer shape; common factor across slot 43
        // (where the divide loop is dynamically dead) and the others is
        // indexing a `&'static [u8; N]` array reference, suggesting the
        // alpha NVPTX backend mishandles the address-space tagging for
        // fixed-size static arrays. Slot 44 (which uses `&'static [u8]`
        // slice for its alphabet) FAILs with wrong bytes but doesn't crash
        // — strong evidence the discriminator is array-ref vs slice.
        //
        // Once any kernel faults the CUDA context is sticky-errored, so
        // run all crashers at the very end to preserve earlier slots'
        // results.
        run_slot!(43, kernel_self_test_base58_all_zeros);
        run_slot!(45, kernel_self_test_bech32_p2wpkh);
        run_slot!(41, kernel_self_test_base58_var_len);
        run_slot!(42, kernel_self_test_base58_var_len_leading_zero);

        report(&results)
    }
}
