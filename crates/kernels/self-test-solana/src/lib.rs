//! Original solana known-answer checks through stock Rust and direct Metal.
#![no_std]

pub use logic::self_test::runners::solana::kernel_self_test_solana;

#[cfg(test)]
mod tests {
    use super::*;
    const SENTINEL: u32 = 0xa5a5a5a5;
    #[test]
    fn original_checks_and_slot_selection_preserve_guards() {
        let count = logic::self_test::SELF_TEST_NUM_CHECKS;
        let cases = logic::self_test::metadata::CASES;
        let owner = "kernel_self_test_solana";
        for selector in [
            u32::MAX,
            cases.iter().find(|c| c.kernel == owner).unwrap().slot as u32,
            count as u32,
        ] {
            let mut results = [SENTINEL; logic::self_test::SELF_TEST_NUM_CHECKS + 2];
            unsafe { kernel_self_test_solana(&selector, results.as_mut_ptr().add(1).cast()) };
            assert_eq!(results[0], SENTINEL);
            assert_eq!(results[count + 1], SENTINEL);
            for case in cases {
                let selected = case.kernel == owner
                    && (selector == u32::MAX || selector as usize == case.slot);
                assert_eq!(
                    results[case.slot + 1],
                    if selected { 1 } else { SENTINEL },
                    "{}",
                    case.name
                );
            }
        }
    }
    #[test]
    fn exported_cases_execute_original_checks_and_only_their_slot() {
        let count = logic::self_test::SELF_TEST_NUM_CHECKS;
        for case in logic::self_test::metadata::CASES
            .iter()
            .filter(|case| case.enabled)
        {
            let entry = logic::self_test::runners::solana::metal_entry(case.slot).unwrap();
            for selector in [case.slot as u32, count as u32] {
                let mut results = [SENTINEL; logic::self_test::SELF_TEST_NUM_CHECKS + 2];
                unsafe {
                    entry(&selector, results.as_mut_ptr().add(1).cast());
                }
                for (index, value) in results.into_iter().enumerate() {
                    assert_eq!(
                        value,
                        if selector == case.slot as u32 && index == case.slot + 1 {
                            1
                        } else {
                            SENTINEL
                        },
                        "{} slot {index}",
                        case.name
                    );
                }
            }
        }
    }
}
