//! Shared test inventory and reporting for CPU, CUDA, and CuMetal.
include!(concat!(env!("OUT_DIR"), "/self_test_entries.rs"));
const _: [(); logic::self_test::SELF_TEST_NUM_CHECKS] = [(); SELF_TEST_ENTRIES.len()];

#[derive(Clone, Copy)]
pub struct Case {
    /// Stable result slot shared by all backends.
    pub slot: usize,
    pub label: &'static str,
    pub kernel: &'static str,
}
/// Standalone PTX containing this entry.
pub fn module_name(kernel: &str) -> &str {
    kernel.strip_prefix("kernel_").expect("kernel entry prefix")
}

pub fn inventory() -> Vec<Case> {
    SELF_TEST_ENTRIES
        .iter()
        .enumerate()
        .filter(|(slot, _)| SELF_TEST_ENABLED[*slot])
        .map(|(slot, &kernel)| Case {
            slot,
            label: logic::self_test::SELF_TEST_LABELS[slot],
            kernel,
        })
        .collect()
}
/// Cache each mode's launch while retaining per-slot reporting.
#[derive(Default)]
pub struct DeviceResults {
    kernels: std::collections::HashMap<&'static str, Result<Vec<u32>, String>>,
}
pub const SENTINEL: u32 = 0xa5a5a5a5;
impl DeviceResults {
    pub fn check(
        &mut self,
        case: Case,
        launch: impl FnOnce() -> Result<Vec<u32>, String>,
    ) -> Result<Outcome, String> {
        let results = self.kernels.entry(case.kernel).or_insert_with(|| {
            let results = launch()?;
            if results.len() != SELF_TEST_ENTRIES.len() {
                return Err("incorrect self-test result length".into());
            }
            for (slot, &value) in results.iter().enumerate() {
                let owned = SELF_TEST_ENTRIES[slot] == case.kernel;
                if !owned && value != SENTINEL {
                    return Err(format!("{} overwrote unrelated slot {slot}", case.kernel));
                }
            }
            Ok(results)
        });
        let results = results.as_ref().map_err(Clone::clone)?;
        let slot = case.slot;
        if results[slot] != 1 {
            return Err(format!(
                "result slot {slot}: got {}, expected 1",
                results[slot]
            ));
        }
        Ok(Outcome::Passed)
    }
}

pub enum Outcome {
    Passed,
    Skipped(&'static str),
}
pub fn run(
    backend: &str,
    mut execute: impl FnMut(Case) -> Result<Outcome, String>,
) -> Result<(), String> {
    let (mut passed, mut failed, mut skipped) = (0, 0, 0);
    for case in inventory() {
        match execute(case) {
            Ok(Outcome::Passed) => {
                passed += 1;
                println!("[{backend}] PASS {}", case.label);
            }
            Ok(Outcome::Skipped(reason)) => {
                skipped += 1;
                println!("[{backend}] SKIP {}: {reason}", case.label);
            }
            Err(error) => {
                failed += 1;
                eprintln!("[{backend}] FAIL {}: {error}", case.label);
            }
        }
    }
    println!("[{backend}] self-test: {passed} passed, {failed} failed, {skipped} skipped");
    if failed == 0 {
        Ok(())
    } else {
        Err(format!("{backend} self-test: {failed} failed"))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_launches_preserve_individual_failures() {
        let mut cache = DeviceResults::default();
        let mut launches = 0;
        for case in inventory() {
            let result = cache.check(case, || {
                launches += 1;
                let mut results = vec![SENTINEL; SELF_TEST_ENTRIES.len()];
                for (slot, &kernel) in SELF_TEST_ENTRIES.iter().enumerate() {
                    if kernel == case.kernel {
                        results[slot] = if slot == 0 { 0 } else { 1 };
                    }
                }
                Ok(results)
            });
            assert_eq!(result.is_err(), case.slot == 0);
        }
        let groups: std::collections::HashSet<_> = inventory().iter().map(|c| c.kernel).collect();
        assert_eq!(launches, groups.len());
    }
    #[test]
    fn unrelated_writes_and_launch_errors_fail_the_group() {
        let cases = inventory();
        let case = cases[0];
        let mut cache = DeviceResults::default();
        assert!(
            cache
                .check(case, || Ok(vec![1; SELF_TEST_ENTRIES.len()]))
                .is_err()
        );
        assert!(
            cache
                .check(case, || panic!("must reuse failed launch"))
                .is_err()
        );
        let mut cache = DeviceResults::default();
        assert!(cache.check(case, || Err("launch failed".into())).is_err());
        assert!(
            cache
                .check(case, || panic!("must reuse failed launch"))
                .is_err()
        );
    }
    #[test]
    fn inventory_preserves_legacy_slots_and_unique_entries() {
        let cases = inventory();
        let mut expected = [SENTINEL; logic::self_test::SELF_TEST_NUM_CHECKS];
        logic::self_test::run_self_test(&mut expected);
        let slots: Vec<_> = expected
            .iter()
            .enumerate()
            .filter_map(|(slot, &value)| (value != SENTINEL).then_some(slot))
            .collect();
        assert!(!slots.is_empty());
        assert_eq!(
            cases.iter().map(|case| case.slot).collect::<Vec<_>>(),
            slots
        );
        for case in cases {
            assert_eq!(expected[case.slot], 1);
        }
    }
    #[test]
    fn failure_is_retained_and_remaining_tests_are_reported() {
        let first = inventory()[0].slot;
        let mut seen = 0;
        let result = run("test", |case| {
            seen += 1;
            if case.slot == first {
                Err("injected failure".into())
            } else {
                Ok(Outcome::Skipped("test backend"))
            }
        });
        assert!(result.is_err());
        assert_eq!(seen, inventory().len());
    }
}
