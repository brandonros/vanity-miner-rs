//! rsa modulus checks; result slot numbers are stable across backends.
use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_self_test_rsa_modulus(results_ptr: *mut u32) {
    let results = unsafe {
        core::slice::from_raw_parts_mut(results_ptr, logic::self_test::SELF_TEST_NUM_CHECKS)
    };
    results[145] = logic::self_test::check_rsa_modulus_multiplication_carry();
    results[146] = logic::self_test::check_rsa_modulus_progression_carry();
    results[147] = logic::self_test::check_rsa_modulus_prime_filter();
    results[148] = logic::self_test::check_rsa_modulus_pseudoprime_rejected();
    results[149] = logic::self_test::check_rsa_modulus_zero_stride_rejected();
    results[150] = logic::self_test::check_rsa_modulus_upper_bound_rejected();
    results[151] = logic::self_test::check_rsa_modulus_equal_factors_rejected();
    results[152] = logic::self_test::check_rsa_modulus_undersized_factor_rejected();
    results[156] = logic::self_test::check_rsa_modulus_end_to_end();
}
