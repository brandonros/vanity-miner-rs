//! p256 public key checks; result slot numbers are stable across backends.
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_p256_public_key(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    results[118] = logic::self_test::check_p256_public_key_hmac_derivation();
    results[119] = logic::self_test::check_p256_public_key_scalar_derivation();
    results[120] = logic::self_test::check_p256_public_key_generator();
    results[121] = logic::self_test::check_p256_public_key_point_double();
    results[122] = logic::self_test::check_p256_public_key_zero_scalar_rejected();
    results[123] = logic::self_test::check_p256_public_key_order_scalar_rejected();
    results[124] = logic::self_test::check_p256_public_key_x_encoding();
    results[125] = logic::self_test::check_p256_public_key_y_encoding();
    results[153] = logic::self_test::check_p256_public_end_to_end();
}
