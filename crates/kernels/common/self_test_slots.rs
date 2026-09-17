// Shared host-side validation of each device entry point's slot ownership.
fn check_slots(kernel_name: &str, kernel: unsafe extern "C" fn(*mut u32)) {
    let kernels = &[(kernel_name, kernel)];
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
