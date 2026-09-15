//! rsa pss checks; result slot numbers are stable across backends.
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_rsa_pss(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    results[135] = logic::self_test::check_rsa_pss_sha256();
    results[136] = logic::self_test::check_rsa_pss_mgf1_partial_block();
    results[137] = logic::self_test::check_rsa_pss_salt32_encoding();
    results[138] = logic::self_test::check_rsa_pss_empty_salt_encoding();
    results[139] = logic::self_test::check_rsa_pss_maximum_salt_encoding();
    results[140] = logic::self_test::check_rsa_pss_oversized_salt_rejected();
    results[141] = logic::self_test::check_rsa_pss_salt_carry();
    results[142] = logic::self_test::check_rsa_pss_crt_known_answer();
    results[143] = logic::self_test::check_rsa_pss_crt_fault_rejected();
    results[144] = logic::self_test::check_rsa_pss_crt_modulus_rejected();
    results[155] = logic::self_test::check_rsa_pss_end_to_end();
}
