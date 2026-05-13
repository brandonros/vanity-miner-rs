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

        // Slots run in strict numeric order. Earlier compiler versions
        // faulted on several alphabet-lookup kernels (41/42/43/45/63) and
        // forced a crash-tail layout; the v1.43 fixes made those slots
        // FAIL-clean or PASS, so the ordering no longer matters for
        // result preservation.
        //
        // Group boundaries (see SELF_TEST_LABELS in logic for details):
        //   0-3   solana per-primitive bisect
        //   4-9   non-solana primitive bisect (k256 / hashes)
        //   10-30 composed-subsystem KATs
        //   31-40 tier-1 arithmetic
        //   41-45 base58 / bech32 var-len + alphabet lookup
        //   46-56 tier-2 arithmetic (dalek/k256 PTX idioms)
        //   57-58 black_box identity probes
        //   59-63 base58 div-by-58 + iter_mut/lookup bisects
        //   64-68 cuda-oxide standalone-repro ports
        //   69    base58 Phase A inner-mutate
        //   70-72 curve25519-dalek per-stage bisect
        //   73-75 k256 per-stage bisect
        //   76-77 unifying-hypothesis probe: `&'static` array of u64s
        //   78-80 k256 Bug-B triangulation (encode, double, Scalar)
        //   81-83 post-v1.46 re-bisect (u128 shr, deep newtype, rev iter)
        //   84-87 dalek Scalar52 ladder bisect of slot 71 chain
        run_slot!(0,  kernel_self_test_primitive_xoroshiro);
        run_slot!(1,  kernel_self_test_primitive_sha512);
        run_slot!(2,  kernel_self_test_primitive_ed25519);
        run_slot!(3,  kernel_self_test_primitive_base58);
        run_slot!(4,  kernel_self_test_primitive_secp256k1_compressed);
        run_slot!(5,  kernel_self_test_primitive_secp256k1_uncompressed);
        run_slot!(6,  kernel_self_test_primitive_keccak256);
        run_slot!(7,  kernel_self_test_primitive_ripemd160);
        run_slot!(8,  kernel_self_test_primitive_sha256_32);
        run_slot!(9,  kernel_self_test_primitive_sha256_variable);
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
        run_slot!(41, kernel_self_test_base58_var_len);
        run_slot!(42, kernel_self_test_base58_var_len_leading_zero);
        run_slot!(43, kernel_self_test_base58_all_zeros);
        run_slot!(44, kernel_self_test_xoroshiro_base64_nonce);
        run_slot!(45, kernel_self_test_bech32_p2wpkh);
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
        run_slot!(57, kernel_self_test_arith_blackbox_identity_u64);
        run_slot!(58, kernel_self_test_arith_blackbox_identity_u32);
        run_slot!(59, kernel_self_test_base58_div_by_58);
        run_slot!(60, kernel_self_test_iter_static_table_lookup);
        run_slot!(61, kernel_self_test_iter_mut_slice_partial);
        run_slot!(62, kernel_self_test_iter_mut_alphabet_lookup);
        run_slot!(63, kernel_self_test_iter_static_slice_lookup);
        run_slot!(64, kernel_self_test_arith_divrem_by_58_pow_5);
        run_slot!(65, kernel_self_test_arith_i128_chain_add);
        run_slot!(66, kernel_self_test_base58_limb_divrem);
        run_slot!(67, kernel_self_test_dynamic_index_write);
        run_slot!(68, kernel_self_test_arith_widening_mul_chain_3term);
        run_slot!(69, kernel_self_test_base58_inner_mutate_phase);
        run_slot!(70, kernel_self_test_dalek_clamp_integer);
        run_slot!(71, kernel_self_test_dalek_scalar_round_trip_one);
        run_slot!(72, kernel_self_test_dalek_mul_base_scalar_one);
        run_slot!(73, kernel_self_test_k256_secret_from_bytes_one);
        run_slot!(74, kernel_self_test_k256_derive_scalar_one);
        run_slot!(75, kernel_self_test_k256_derive_scalar_two);
        run_slot!(76, kernel_self_test_static_u64_array_lookup);
        run_slot!(77, kernel_self_test_static_struct_wrapped_u64_lookup);
        run_slot!(78, kernel_self_test_k256_encode_generator);
        run_slot!(79, kernel_self_test_k256_double_generator);
        run_slot!(80, kernel_self_test_k256_scalar_one_round_trip);
        run_slot!(81, kernel_self_test_arith_u128_imm_shr_52);
        run_slot!(82, kernel_self_test_static_depth4_newtype_nesting);
        run_slot!(83, kernel_self_test_reverse_range_write);
        run_slot!(84, kernel_self_test_dalek_scalar52_from_bytes);
        run_slot!(85, kernel_self_test_dalek_scalar52_montgomery_reduce_r);
        run_slot!(86, kernel_self_test_dalek_scalar52_mul_internal_then_reduce_one_r);
        run_slot!(87, kernel_self_test_dalek_scalar52_as_bytes_one);

        report(&results)
    }
}
