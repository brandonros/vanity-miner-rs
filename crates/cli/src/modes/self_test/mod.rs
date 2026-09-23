//! Check inventory and reporting shared by the CPU and CUDA runners.
pub mod args;
use logic::self_test::{MODES, Mode};
use std::collections::HashMap;

/// One check of one compiled-in mode.
#[derive(Clone, Debug)]
pub struct Case {
    pub mode: &'static Mode,
    pub index: usize,
    /// `<mode>.<check>`, the `--check` selector.
    pub name: String,
}

impl Case {
    pub fn label(&self) -> &'static str {
        self.mode.checks[self.index].label
    }
}

pub fn inventory() -> Vec<Case> {
    MODES
        .iter()
        .flat_map(|mode| {
            mode.checks
                .iter()
                .enumerate()
                .map(move |(index, check)| Case {
                    mode,
                    index,
                    name: format!("{}.{}", mode.name, check.name),
                })
        })
        .collect()
}

/// Result buffers start as SENTINEL so an unwritten check fails.
pub const SENTINEL: u32 = 0xa5a5a5a5;

/// Runs each selected mode once and reports its selected checks.
pub fn run(
    backend: &str,
    cases: &[Case],
    mut run_mode: impl FnMut(&'static Mode) -> Result<Vec<u32>, String>,
) -> Result<(), String> {
    let mut launches: HashMap<&str, Result<Vec<u32>, String>> = HashMap::new();
    let (mut passed, mut failed) = (0, 0);
    for case in cases {
        let results = launches.entry(case.mode.name).or_insert_with(|| {
            let results = run_mode(case.mode)?;
            if results.len() != case.mode.checks.len() {
                return Err(format!(
                    "{} wrote {} results, expected {}",
                    case.mode.name,
                    results.len(),
                    case.mode.checks.len()
                ));
            }
            Ok(results)
        });
        let outcome = match results {
            Ok(results) if results[case.index] == 1 => Ok(()),
            Ok(results) => Err(format!("got {}, expected 1", results[case.index])),
            Err(error) => Err(error.clone()),
        };
        match outcome {
            Ok(()) => {
                passed += 1;
                println!("[{backend}] PASS {}", case.name);
            }
            Err(error) => {
                failed += 1;
                eprintln!("[{backend}] FAIL {}: {error}", case.name);
            }
        }
    }
    println!("[{backend}] self-test: {passed} passed, {failed} failed");
    if failed == 0 {
        Ok(())
    } else {
        Err(format!("{backend} self-test: {failed} failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn failures(result: Result<(), String>) -> usize {
        match result {
            Ok(()) => 0,
            Err(error) => error
                .strip_prefix("test self-test: ")
                .and_then(|rest| rest.strip_suffix(" failed"))
                .and_then(|count| count.parse().ok())
                .unwrap_or_else(|| panic!("unexpected error: {error}")),
        }
    }

    #[test]
    fn inventory_names_are_unique_and_cover_every_check() {
        let cases = inventory();
        let total: usize = MODES.iter().map(|mode| mode.checks.len()).sum();
        assert_eq!(cases.len(), total);
        assert!(!cases.is_empty());
        for (index, case) in cases.iter().enumerate() {
            assert!(cases[..index].iter().all(|other| other.name != case.name));
            assert_eq!(
                case.name,
                format!("{}.{}", case.mode.name, case.mode.checks[case.index].name)
            );
            assert!(!case.label().is_empty());
        }
    }

    #[test]
    fn each_mode_runs_once_and_one_bad_result_fails_one_check() {
        let cases = inventory();
        let mut launches = 0;
        let result = run("test", &cases, |mode| {
            launches += 1;
            let mut results = vec![1; mode.checks.len()];
            if mode.name == cases[0].mode.name {
                results[0] = 0;
            }
            Ok(results)
        });
        assert_eq!(failures(result), 1);
        assert_eq!(launches, MODES.len());
        assert_eq!(
            failures(run("test", &cases, |mode| Ok(vec![1; mode.checks.len()]))),
            0
        );
    }

    #[test]
    fn launch_errors_fail_every_check_of_that_mode_only() {
        let cases = inventory();
        let broken = cases[0].mode;
        let mut launches = 0;
        let result = run("test", &cases, |mode| {
            launches += 1;
            if mode.name == broken.name {
                Err("launch failed".into())
            } else {
                Ok(vec![1; mode.checks.len()])
            }
        });
        assert_eq!(failures(result), broken.checks.len());
        assert_eq!(launches, MODES.len());
    }

    #[test]
    fn wrong_result_counts_and_unexpected_values_fail() {
        let cases = inventory();
        let mode = cases[0].mode;
        for length in [mode.checks.len() - 1, mode.checks.len() + 1] {
            let result = run("test", &cases, |launched| {
                let count = if launched.name == mode.name {
                    length
                } else {
                    launched.checks.len()
                };
                Ok(vec![1; count])
            });
            assert_eq!(failures(result), mode.checks.len());
        }
        for value in [0, 2, SENTINEL, u32::MAX] {
            let result = run("test", &cases, |launched| {
                let mut results = vec![1; launched.checks.len()];
                if launched.name == mode.name {
                    results[0] = value;
                }
                Ok(results)
            });
            assert_eq!(
                failures(result),
                1,
                "unexpected result {value} must not pass"
            );
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;
