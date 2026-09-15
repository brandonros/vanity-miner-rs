//! bitcoin checks; result slot numbers are stable across backends.
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_bitcoin(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    results[4] = logic::self_test::check_primitive_secp256k1_compressed();
    results[7] = logic::self_test::check_primitive_ripemd160();
    results[8] = logic::self_test::check_primitive_sha256_32();
    results[16] = logic::self_test::check_bitcoin_priv();
    results[17] = logic::self_test::check_bitcoin_pub();
    results[18] = logic::self_test::check_bitcoin_pkh();
    results[19] = logic::self_test::check_bitcoin_encoded();
    results[20] = logic::self_test::check_bitcoin_matches();
    results[21] = logic::self_test::check_wif_compressed_mainnet();
    results[22] = logic::self_test::check_wif_uncompressed_mainnet();
    results[23] = logic::self_test::check_wif_compressed_testnet();
    results[24] = logic::self_test::check_wif_uncompressed_testnet();
    results[42] = logic::self_test::check_base58_var_len_leading_zero();
    results[45] = logic::self_test::check_bech32_p2wpkh();
    results[73] = logic::self_test::check_k256_secret_from_bytes_one();
    results[74] = logic::self_test::check_k256_derive_scalar_one();
    results[75] = logic::self_test::check_k256_derive_scalar_two();
    results[76] = logic::self_test::check_static_u64_array_lookup();
    results[77] = logic::self_test::check_static_struct_wrapped_u64_lookup();
    results[78] = logic::self_test::check_k256_encode_generator();
    results[79] = logic::self_test::check_k256_double_generator();
    results[80] = logic::self_test::check_k256_scalar_one_round_trip();
    results[93] = logic::self_test::check_k256_affine_generator_encode();
    results[94] = logic::self_test::check_subtle_choice_u8_into_bool();
    results[95] = logic::self_test::check_subtle_conditional_select_u64();
    results[96] = logic::self_test::check_k256_encoded_point_from_affine_coords();
    results[97] = logic::self_test::check_index_trait_const_indices();
    results[98] = logic::self_test::check_generic_array_basic_index();
    results[99] = logic::self_test::check_generic_array_copy_from_slice();
    results[100] = logic::self_test::check_from_affine_coords_replica();
    results[101] = logic::self_test::check_generic_array_as_slice_last();
    results[104] = logic::self_test::check_field_bytes_into_conversion();
    results[110] = logic::self_test::check_generic_array_copy_from_ga_source();
}
