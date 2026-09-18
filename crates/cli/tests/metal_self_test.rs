#![cfg(all(feature = "metal", feature = "self_test_support", target_os = "macos"))]
use std::{path::PathBuf, process::Command};
fn artifacts() -> PathBuf {
    std::env::var_os("VANITY_METAL_SELF_TEST_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/metal"))
}
fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_vanity-miner"));
    command
        .arg("--metal-artifacts")
        .arg(artifacts())
        .arg("self-test");
    command
}
#[test]
#[ignore = "build all enabled self-test Metal bundles first; requires Apple GPU"]
fn original_registry_groups_pass_on_metal_without_skips() {
    let output = command().output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cases: Vec<_> = logic::self_test::metadata::CASES
        .iter()
        .filter(|case| case.enabled)
        .collect();
    for case in &cases {
        assert!(
            stdout
                .lines()
                .any(|line| line == format!("[Metal] PASS {}", case.name)),
            "missing {}",
            case.name
        );
    }
    assert!(stdout.contains(&format!(
        "[Metal] self-test: {} passed, 0 failed, 0 skipped",
        cases.len()
    )));
    assert!(!stdout.contains("[CPU]"));
}
#[test]
#[ignore = "build all enabled self-test Metal bundles first; requires Apple GPU"]
fn named_checks_use_device_slot_selection_and_registry_order() {
    let cases: Vec<_> = logic::self_test::metadata::CASES
        .iter()
        .filter(|case| case.enabled)
        .collect();
    let first = cases.first().unwrap();
    let last = cases.last().unwrap();
    let output = command()
        .args([
            "--check", last.name, "--check", first.name, "--check", first.name,
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let names: Vec<_> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("[Metal] PASS "))
        .collect();
    assert_eq!(names, vec![first.name, last.name]);
    assert!(stdout.contains("2 passed, 0 failed, 0 skipped"));
}
#[test]
fn list_and_invalid_selection_do_not_require_artifacts() {
    let output = command().arg("--list").output().unwrap();
    assert!(output.status.success());
    let output = command()
        .args(["--check", "unknown.check"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown self-test"));
}

#[cfg(feature = "self_test_shallenge")]
#[test]
#[ignore = "build the Shallenge self-test Metal bundle first; requires Apple GPU"]
fn named_checks_report_load_pipeline_and_gpu_timings() {
    let checks = [
        "shallenge.primitive_sha256_variable",
        "shallenge.sha256_padding_55",
    ];
    let output = command()
        .args(["--check", checks[0], "--check", checks[1]])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("{stderr}");
    assert!(output.status.success(), "{stdout}\n{stderr}");
    for check in checks {
        assert!(stdout.contains(&format!("[Metal] PASS {check}")));
    }
    assert!(stdout.contains("2 passed, 0 failed, 0 skipped"));
    let loads: Vec<_> = stderr
        .lines()
        .filter(|line| line.starts_with("Metal device:"))
        .collect();
    assert_eq!(loads.len(), 2, "{stderr}");
    for load in loads {
        assert!(
            load.contains("library load ") && load.contains("pipeline creation "),
            "{load}"
        );
    }
    let dispatches: Vec<_> = stderr
        .lines()
        .filter(|line| line.starts_with("[Metal] dispatch "))
        .collect();
    assert_eq!(dispatches.len(), 2, "{stderr}");
    for dispatch in dispatches {
        assert!(
            dispatch.contains(": GPU ") && dispatch.contains("dispatch wall "),
            "{dispatch}"
        );
    }
    let summaries: Vec<_> = stderr
        .lines()
        .filter(|line| line.starts_with("[Metal] self-test timings:"))
        .collect();
    assert_eq!(summaries.len(), 1, "{stderr}");
    let summary = summaries[0];
    for field in [
        "library load ",
        "pipeline creation ",
        "dispatch wall ",
        "GPU ",
        "/2 launches timed)",
        "2 pipelines loaded",
    ] {
        assert!(summary.contains(field), "{summary}");
    }
}
