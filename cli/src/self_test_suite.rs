//! Shared test inventory and reporting for CPU, CUDA, and CuMetal.
include!(concat!(env!("OUT_DIR"), "/self_test_entries.rs"));
const _: [(); logic::self_test::SELF_TEST_NUM_CHECKS] = [(); SELF_TEST_ENTRIES.len()];

#[derive(Clone, Copy)]
pub struct Case {
    /// None identifies the launch probe; every other case has one result slot.
    pub slot: Option<usize>,
    pub label: &'static str,
    pub kernel: &'static str,
}
pub fn inventory() -> Vec<Case> {
    let mut cases = vec![Case {
        slot: None,
        label: "launch probe",
        kernel: "kernel_self_test_stub",
    }];
    for (slot, &kernel) in SELF_TEST_ENTRIES.iter().enumerate() {
        cases.push(Case {
            slot: Some(slot),
            label: logic::self_test::SELF_TEST_LABELS[slot],
            kernel,
        });
    }
    cases
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
    fn inventory_preserves_legacy_slots_and_unique_entries() {
        let cases = inventory();
        let legacy: Vec<_> = cases.iter().filter_map(|case| case.slot).collect();
        assert_eq!(
            legacy,
            (0..logic::self_test::SELF_TEST_NUM_CHECKS).collect::<Vec<_>>()
        );
        let names: std::collections::HashSet<_> = cases.iter().map(|c| c.kernel).collect();
        assert_eq!(names.len(), cases.len());
    }
    #[test]
    fn failure_is_retained_and_remaining_tests_are_reported() {
        let mut seen = 0;
        let result = run("test", |case| {
            seen += 1;
            if case.slot == Some(0) {
                Err("injected failure".into())
            } else {
                Ok(Outcome::Skipped("test backend"))
            }
        });
        assert!(result.is_err());
        assert_eq!(seen, inventory().len());
    }
}
