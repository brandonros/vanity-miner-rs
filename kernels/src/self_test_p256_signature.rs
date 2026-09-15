//! p256 signature checks; result slot numbers are stable across backends.
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_p256_signature(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    results[126] = logic::self_test::check_p256_signature_rfc6979_sample();
    results[127] = logic::self_test::check_p256_signature_rfc6979_test();
    results[128] = logic::self_test::check_p256_signature_ephemeral_r();
    results[129] = logic::self_test::check_p256_signature_ephemeral_signature();
    results[130] = logic::self_test::check_p256_signature_zero_nonce_rejected();
    results[131] = logic::self_test::check_p256_signature_low_s();
    results[132] = logic::self_test::check_p256_signature_high_s();
    results[133] = logic::self_test::check_p256_signature_message_window_carry();
    results[134] = logic::self_test::check_p256_signature_ephemeral_hmac();
    results[154] = logic::self_test::check_p256_signature_end_to_end();
}
