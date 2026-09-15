//! solana checks; result slot numbers are stable across backends.
//! Shared integer and indexing regressions live here alongside Base58 and Dalek.
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_solana(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    results[0] = logic::self_test::check_primitive_xoroshiro();
    results[1] = logic::self_test::check_primitive_sha512();
    results[2] = logic::self_test::check_primitive_ed25519();
    results[3] = logic::self_test::check_primitive_base58();
    results[10] = logic::self_test::check_solana_priv();
    results[11] = logic::self_test::check_solana_pub();
    results[12] = logic::self_test::check_solana_encoded();
    results[31] = logic::self_test::check_arith_u32_div_var();
    results[32] = logic::self_test::check_arith_u32_div_const();
    results[33] = logic::self_test::check_arith_u64_div_var();
    results[34] = logic::self_test::check_arith_u64_div_const();
    results[35] = logic::self_test::check_arith_u32_rem_var();
    results[36] = logic::self_test::check_arith_u64_rem_var();
    results[37] = logic::self_test::check_arith_u32_mul_lo();
    results[38] = logic::self_test::check_arith_u64_mul_lo();
    results[39] = logic::self_test::check_arith_u64_mul_hi();
    results[40] = logic::self_test::check_arith_u128_mul();
    results[41] = logic::self_test::check_base58_var_len();
    results[43] = logic::self_test::check_base58_all_zeros();
    results[46] = logic::self_test::check_arith_overflowing_add();
    results[47] = logic::self_test::check_arith_overflowing_sub();
    results[48] = logic::self_test::check_arith_carry_chain_3limb();
    results[49] = logic::self_test::check_arith_widening_mul_pair();
    results[50] = logic::self_test::check_arith_mad_lo_u64();
    results[51] = logic::self_test::check_arith_mad_hi_u64();
    results[52] = logic::self_test::check_arith_mul_wide_u32();
    results[53] = logic::self_test::check_arith_mask_blend_true();
    results[54] = logic::self_test::check_arith_mask_blend_false();
    results[55] = logic::self_test::check_arith_var_shr_u64();
    results[56] = logic::self_test::check_arith_var_shl_u64();
    results[57] = logic::self_test::check_arith_blackbox_identity_u64();
    results[58] = logic::self_test::check_arith_blackbox_identity_u32();
    results[59] = logic::self_test::check_base58_div_by_58();
    results[60] = logic::self_test::check_iter_static_table_lookup();
    results[61] = logic::self_test::check_iter_mut_slice_partial();
    results[62] = logic::self_test::check_iter_mut_alphabet_lookup();
    results[63] = logic::self_test::check_iter_static_slice_lookup();
    results[64] = logic::self_test::check_arith_divrem_by_58_pow_5();
    results[65] = logic::self_test::check_arith_i128_chain_add();
    results[66] = logic::self_test::check_base58_limb_divrem();
    results[67] = logic::self_test::check_dynamic_index_write();
    results[68] = logic::self_test::check_arith_widening_mul_chain_3term();
    results[69] = logic::self_test::check_base58_inner_mutate_phase();
    results[70] = logic::self_test::check_dalek_clamp_integer();
    results[71] = logic::self_test::check_dalek_scalar_round_trip_one();
    results[72] = logic::self_test::check_dalek_mul_base_scalar_one();
    results[81] = logic::self_test::check_arith_u128_imm_shr_52();
    results[82] = logic::self_test::check_static_depth4_newtype_nesting();
    results[83] = logic::self_test::check_reverse_range_write();
    results[84] = logic::self_test::check_dalek_scalar52_from_bytes();
    results[85] = logic::self_test::check_dalek_scalar52_montgomery_reduce_r();
    results[86] = logic::self_test::check_dalek_scalar52_mul_internal_then_reduce_one_r();
    results[87] = logic::self_test::check_dalek_scalar52_as_bytes_one();
    results[88] = logic::self_test::check_dalek_scalar52_sub_no_underflow();
    results[89] = logic::self_test::check_dalek_scalar52_sub_with_underflow();
    results[90] = logic::self_test::check_dalek_scalar52_montgomery_reduce_with_sub();
    results[91] = logic::self_test::check_index_trait_dispatch();
    results[92] = logic::self_test::check_dalek_scalar_one_to_bytes_direct();
    results[102] = logic::self_test::check_dalek_scalar_round_trip_zero();
    results[103] = logic::self_test::check_dalek_scalar_from_bytes_wide_zero();
    results[105] = logic::self_test::check_base58_min_nonzero();
    results[106] = logic::self_test::check_named_field_struct_return();
    results[107] = logic::self_test::check_base58_handrolled_no_seq();
    results[108] = logic::self_test::check_slice_reverse_partial();
    results[109] = logic::self_test::check_dalek_scalar_eq_zero();
    results[111] = logic::self_test::check_dalek_zero_eq_zero();
    results[112] = logic::self_test::check_dalek_from_canonical_zero();
    results[113] = logic::self_test::check_dalek_scalar52_from_bytes_zero();
    results[114] = logic::self_test::check_dalek_scalar52_mul_internal_zero();
    results[115] = logic::self_test::check_dalek_scalar52_montgomery_reduce_zero();
    results[116] = logic::self_test::check_dalek_scalar52_as_bytes_zero();
    results[117] = logic::self_test::check_dalek_reduce_pipeline_zero();
}
