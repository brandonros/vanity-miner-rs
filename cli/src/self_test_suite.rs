//! Shared test inventory and reporting for CPU, CUDA, and CuMetal.
include!(concat!(env!("OUT_DIR"), "/self_test_entries.rs"));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Probe,
    Legacy(usize),
    P256Public,
    P256Signature,
    RsaPss,
    RsaModulus,
}
#[derive(Clone, Copy)]
pub struct Case {
    pub kind: Kind,
    pub label: &'static str,
    pub kernel: &'static str,
}
pub fn inventory() -> Vec<Case> {
    let mut cases = vec![Case {
        kind: Kind::Probe,
        label: "launch probe",
        kernel: "kernel_self_test_stub",
    }];
    for (slot, &kernel) in SELF_TEST_ENTRIES.iter().enumerate() {
        cases.push(Case {
            kind: Kind::Legacy(slot),
            label: logic::SELF_TEST_LABELS[slot],
            kernel,
        });
    }
    #[cfg(feature = "p256-public-key")]
    cases.push(Case {
        kind: Kind::P256Public,
        label: "P-256 public point targets and counter carries",
        kernel: "kernel_p256_public_key_vanity",
    });
    #[cfg(feature = "p256-signature")]
    cases.push(Case {
        kind: Kind::P256Signature,
        label: "P-256 message/ephemeral signatures and S forms",
        kernel: "kernel_p256_signature_vanity",
    });
    #[cfg(feature = "rsa-pss")]
    cases.push(Case {
        kind: Kind::RsaPss,
        label: "RSA CRT, PSS salt boundaries and carry propagation",
        kernel: "kernel_rsa_pss_signature_vanity",
    });
    #[cfg(feature = "rsa-modulus")]
    cases.push(Case {
        kind: Kind::RsaModulus,
        label: "RSA progression, primality and upper bounds",
        kernel: "kernel_rsa_modulus_vanity",
    });
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
        let legacy: Vec<_> = cases
            .iter()
            .filter_map(|case| {
                if let Kind::Legacy(slot) = case.kind {
                    Some(slot)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(legacy, (0..logic::SELF_TEST_NUM_CHECKS).collect::<Vec<_>>());
        let names: std::collections::HashSet<_> = cases.iter().map(|c| c.kernel).collect();
        assert_eq!(names.len(), cases.len());
    }
    #[test]
    fn failure_is_retained_and_remaining_tests_are_reported() {
        let mut seen = 0;
        let result = run("test", |case| {
            seen += 1;
            if case.kind == Kind::Legacy(0) {
                Err("injected failure".into())
            } else {
                Ok(Outcome::Skipped("test backend"))
            }
        });
        assert!(result.is_err());
        assert_eq!(seen, inventory().len());
    }
}
