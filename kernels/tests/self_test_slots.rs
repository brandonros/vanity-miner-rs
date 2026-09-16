#![cfg(any(
    feature = "self_test_solana",
    feature = "self_test_bitcoin",
    feature = "self_test_ethereum",
    feature = "self_test_shallenge",
    feature = "self_test_p256_public_key",
    feature = "self_test_p256_signature",
    feature = "self_test_rsa_pss",
    feature = "self_test_rsa_modulus"
))]
mod tests {
    #[test]
    fn mode_kernels_write_every_slot_once_without_touching_guards() {
        let kernels: &[(&str, unsafe extern "C" fn(*mut u32))] = &[
            #[cfg(feature = "self_test_solana")]
            (
                "kernel_self_test_solana",
                kernels::self_test::solana::kernel_self_test_solana,
            ),
            #[cfg(feature = "self_test_bitcoin")]
            (
                "kernel_self_test_bitcoin",
                kernels::self_test::bitcoin::kernel_self_test_bitcoin,
            ),
            #[cfg(feature = "self_test_ethereum")]
            (
                "kernel_self_test_ethereum",
                kernels::self_test::ethereum::kernel_self_test_ethereum,
            ),
            #[cfg(feature = "self_test_shallenge")]
            (
                "kernel_self_test_shallenge",
                kernels::self_test::shallenge::kernel_self_test_shallenge,
            ),
            #[cfg(feature = "self_test_p256_public_key")]
            (
                "kernel_self_test_p256_public_key",
                kernels::self_test::p256_public_key::kernel_self_test_p256_public_key,
            ),
            #[cfg(feature = "self_test_p256_signature")]
            (
                "kernel_self_test_p256_signature",
                kernels::self_test::p256_signature::kernel_self_test_p256_signature,
            ),
            #[cfg(feature = "self_test_rsa_pss")]
            (
                "kernel_self_test_rsa_pss",
                kernels::self_test::rsa_pss::kernel_self_test_rsa_pss,
            ),
            #[cfg(feature = "self_test_rsa_modulus")]
            (
                "kernel_self_test_rsa_modulus",
                kernels::self_test::rsa_modulus::kernel_self_test_rsa_modulus,
            ),
        ];
        let mut seen = [false; logic::self_test::SELF_TEST_NUM_CHECKS];
        for &(kernel_name, kernel) in kernels {
            let mut guarded = [0xa5a5a5a5; logic::self_test::SELF_TEST_NUM_CHECKS + 2];
            unsafe { kernel(guarded.as_mut_ptr().add(1)) };
            assert_eq!(guarded[0], 0xa5a5a5a5);
            assert_eq!(*guarded.last().unwrap(), 0xa5a5a5a5);
            let mut written = 0;
            for (slot, &result) in guarded[1..guarded.len() - 1].iter().enumerate() {
                let case = logic::self_test::metadata::CASES
                    .iter()
                    .find(|case| case.slot == slot)
                    .unwrap();
                assert_eq!(
                    result != 0xa5a5a5a5,
                    case.kernel == kernel_name,
                    "kernel {kernel_name} wrote the wrong set of slots at {slot}"
                );
                if result != 0xa5a5a5a5 {
                    assert_eq!(
                        result,
                        if case.gpu_skip.is_some() { 2 } else { 1 },
                        "failed slot {slot}"
                    );
                    assert!(!seen[slot], "duplicate slot {slot}");
                    seen[slot] = true;
                    written += 1;
                }
            }
            assert!(written > 0);
        }
        let mut expected = [0xa5a5a5a5; logic::self_test::SELF_TEST_NUM_CHECKS];
        logic::self_test::run_self_test(&mut expected);
        for (slot, &written) in seen.iter().enumerate() {
            assert_eq!(
                written,
                expected[slot] != 0xa5a5a5a5,
                "slot {slot} feature mismatch"
            );
        }
    }
}
