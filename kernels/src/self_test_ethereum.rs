//! ethereum checks; result slot numbers are stable across backends.
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_ethereum(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    results[5] = logic::self_test::check_primitive_secp256k1_uncompressed();
    results[6] = logic::self_test::check_primitive_keccak256();
    results[13] = logic::self_test::check_ethereum_priv();
    results[14] = logic::self_test::check_ethereum_pub();
    results[15] = logic::self_test::check_ethereum_address();
}
