//! shallenge checks; result slot numbers are stable across backends.
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_shallenge(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    results[9] = logic::self_test::check_primitive_sha256_variable();
    results[25] = logic::self_test::check_shallenge_hash();
    results[26] = logic::self_test::check_shallenge_nonce_len();
    results[27] = logic::self_test::check_shallenge_is_better();
    results[28] = logic::self_test::check_compare_hashes_lt();
    results[29] = logic::self_test::check_compare_hashes_gt();
    results[30] = logic::self_test::check_compare_hashes_eq();
    results[44] = logic::self_test::check_xoroshiro_base64_nonce();
}
