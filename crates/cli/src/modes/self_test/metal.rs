//! Execute the original registry checks through independently built Metal groups.
use crate::runner::{
    RunResult,
    metal::{MetalRunner, transport::load_artifact},
};
use llvm_metal_runtime::Buffer;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

const GUARD: usize = 256;
fn interface(kernel: &str) -> Result<&'static str, String> {
    match kernel {
        "kernel_self_test_solana" => Ok(include_str!(
            "../../../../kernels/self-test-solana/metal/kernel.interface.json"
        )),
        "kernel_self_test_bitcoin" => Ok(include_str!(
            "../../../../kernels/self-test-bitcoin/metal/kernel.interface.json"
        )),
        "kernel_self_test_ethereum" => Ok(include_str!(
            "../../../../kernels/self-test-ethereum/metal/kernel.interface.json"
        )),
        "kernel_self_test_shallenge" => Ok(include_str!(
            "../../../../kernels/self-test-shallenge/metal/kernel.interface.json"
        )),
        "kernel_self_test_p256_public_key" => Ok(include_str!(
            "../../../../kernels/self-test-p256-public-key/metal/kernel.interface.json"
        )),
        "kernel_self_test_p256_signature" => Ok(include_str!(
            "../../../../kernels/self-test-p256-signature/metal/kernel.interface.json"
        )),
        "kernel_self_test_rsa_pss" => Ok(include_str!(
            "../../../../kernels/self-test-rsa-pss/metal/kernel.interface.json"
        )),
        "kernel_self_test_rsa_modulus" => Ok(include_str!(
            "../../../../kernels/self-test-rsa-modulus/metal/kernel.interface.json"
        )),
        _ => Err(format!("unknown Metal self-test group: {kernel}")),
    }
}

fn guarded(size: usize) -> Buffer {
    Buffer {
        bytes: vec![0xa5; GUARD * 2 + size],
        offset: GUARD,
    }
}

fn output(buffer: &Buffer) -> Result<Vec<u32>, String> {
    let size = logic::self_test::SELF_TEST_NUM_CHECKS * 4;
    if buffer.bytes[..GUARD]
        .iter()
        .chain(buffer.bytes[GUARD + size..].iter())
        .any(|&b| b != 0xa5)
    {
        return Err("Metal self-test changed an output guard".into());
    }
    Ok(buffer.bytes[GUARD..GUARD + size]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

fn launch(
    directory: &Path,
    kernel_name: &str,
    selected: Option<&[usize]>,
    case_entry: Option<&str>,
) -> Result<Vec<u32>, String> {
    let json = interface(kernel_name)?;
    let mut declared: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    if let Some(entry) = case_entry {
        declared["entry"] = entry.into();
    }
    let json = declared.to_string();
    if declared["arguments"][1]["bytes"] != logic::self_test::SELF_TEST_NUM_CHECKS * 4 {
        return Err("Metal self-test interface differs from the registry size".into());
    }
    let (kernel, _) = load_artifact(directory, &json)?;
    let mut buffers = [
        guarded(4),
        guarded(logic::self_test::SELF_TEST_NUM_CHECKS * 4),
    ];
    let mut kernel = kernel.prepare(&buffers)?;
    let selectors: Vec<u32> = match selected {
        Some(slots) => slots.iter().map(|&slot| slot as u32).collect(),
        None => vec![u32::MAX],
    };
    let mut results = vec![super::SENTINEL; logic::self_test::SELF_TEST_NUM_CHECKS];
    for selector in selectors {
        buffers[0].bytes[GUARD..GUARD + 4].copy_from_slice(&selector.to_le_bytes());
        let expected_input = buffers[0].bytes.clone();
        // SAFETY: hash/ABI checked single-invocation kernel; disjoint initialized
        // selector and full registry storage, with guards outside both bindings.
        unsafe {
            kernel.run(&mut buffers, 1, 1)?;
        }
        if buffers[0].bytes != expected_input {
            return Err("Metal self-test changed selector input or its guards".into());
        }
        let actual = output(&buffers[1])?;
        for case in logic::self_test::metadata::CASES {
            let writable = case.kernel == kernel_name
                && (selector == u32::MAX || selector as usize == case.slot);
            if !writable && actual[case.slot] != results[case.slot] {
                return Err(format!(
                    "{kernel_name} overwrote unselected check {}",
                    case.name
                ));
            }
            if writable && actual[case.slot] == 2 {
                return Err(format!(
                    "Metal self-test must execute {}, not skip it",
                    case.name
                ));
            }
        }
        results = actual;
    }
    Ok(results)
}

fn directory(runner: &MetalRunner, kernel: &str) -> PathBuf {
    let mode = kernel.strip_prefix("kernel_").unwrap().replace('_', "-");
    match &runner.options.metal_artifacts {
        Some(root)
            if !root.join("kernel.build.json").is_file()
                && !root.join("self-tests.json").is_file()
                && !root.join("self-tests.pending.json").is_file() =>
        {
            root.join(mode)
        }
        _ => runner.artifacts(&mode),
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn validate_case_manifest(
    directory: &Path,
    case: super::Case,
) -> Result<serde_json::Value, String> {
    let manifest = read_json(&directory.join("kernel.build.json"))?;
    let identity = &manifest["self_test"];
    if manifest["schema"] != 1
        || identity["name"] != case.name
        || identity["slot"] != case.slot
        || identity["entry"] != case.metal_entry
        || identity["kernel"] != case.kernel
    {
        return Err(format!(
            "Metal self-test artifact does not identify {}",
            case.name
        ));
    }
    Ok(manifest)
}

/// Require the entire original group inventory, even for a selected-check run.
/// Individual debug bundles use the separate explicit --metal-artifacts path.
fn validate_group(directory: &Path, kernel: &str) -> Result<(), String> {
    let group = read_json(&directory.join("self-tests.json"))?;
    if group["schema"] != 1 || group["kind"] != "self-test-group" || group["kernel"] != kernel {
        return Err("invalid Metal self-test group manifest".into());
    }
    let entries = group["cases"]
        .as_array()
        .ok_or("missing Metal self-test case inventory")?;
    let expected: Vec<_> = logic::self_test::metadata::CASES
        .iter()
        .filter(|case| case.kernel == kernel)
        .collect();
    if entries.len() != expected.len() {
        return Err(format!(
            "incomplete Metal self-test group {kernel}: expected {} cases, got {}",
            expected.len(),
            entries.len()
        ));
    }
    for (entry, case) in entries.iter().zip(expected) {
        let relative = format!("cases/{}", case.name);
        if entry["name"] != case.name
            || entry["slot"] != case.slot
            || entry["entry"] != case.metal_entry
            || entry["directory"] != relative
        {
            return Err(format!(
                "Metal self-test inventory differs at {}",
                case.name
            ));
        }
        let path = directory.join(&relative);
        let bytes = fs::read(path.join("kernel.build.json")).map_err(|e| e.to_string())?;
        if entry["manifest_sha256"].as_str() != Some(hex::encode(Sha256::digest(&bytes)).as_str()) {
            return Err(format!(
                "Metal self-test manifest hash mismatch: {}",
                case.name
            ));
        }
        let manifest = validate_case_manifest(&path, *case)?;
        for field in [
            "source_revision",
            "source_sha256",
            "compiler_sha256",
            "rustc",
        ] {
            if manifest[field].is_null() || manifest[field] != group[field] {
                return Err(format!(
                    "Metal self-test provenance differs for {}: {field}",
                    case.name
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn run(runner: &MetalRunner, args: &super::args::SelfTestArgs) -> RunResult {
    let cases = args.selected()?;
    let mut legacy_cache = super::DeviceResults::default();
    let mut groups = std::collections::HashMap::<&str, Result<(), String>>::new();
    super::run("Metal", &cases, |case| {
        let group_dir = directory(runner, case.kernel);
        if group_dir.join("self-tests.pending.json").exists() {
            return Err("Metal self-test group build is incomplete; rerun the builder".into());
        }
        let (case_dir, is_case) = if group_dir.join("self-tests.json").is_file() {
            groups
                .entry(case.kernel)
                .or_insert_with(|| validate_group(&group_dir, case.kernel))
                .as_ref()
                .map_err(Clone::clone)?;
            (group_dir.join("cases").join(case.name), true)
        } else {
            let manifest = read_json(&group_dir.join("kernel.build.json"))?;
            let is_case = !manifest["self_test"].is_null();
            if is_case && (cases.len() != 1 || args.checks.is_empty()) {
                return Err(
                    "an individual Metal self-test bundle requires exactly one named --check"
                        .into(),
                );
            }
            (group_dir, is_case)
        };
        if is_case {
            validate_case_manifest(&case_dir, case)?;
            let results = launch(
                &case_dir,
                case.kernel,
                Some(&[case.slot]),
                Some(case.metal_entry),
            )?;
            // A fresh per-case cache preserves original outcome validation while
            // allowing each result/error to be reported before the next load.
            super::DeviceResults::default().check(case, || Ok(results))
        } else {
            legacy_cache.check(case, || {
                let slots: Vec<_> = cases
                    .iter()
                    .filter(|owner| owner.kernel == case.kernel)
                    .map(|owner| owner.slot)
                    .collect();
                launch(
                    &case_dir,
                    case.kernel,
                    (!args.checks.is_empty()).then_some(slots.as_slice()),
                    None,
                )
            })
        }
    })
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_registered_groups_have_matching_interfaces() {
        for case in logic::self_test::metadata::CASES {
            let interface: llvm_metal_abi::KernelInterface =
                serde_json::from_str(interface(case.kernel).unwrap()).unwrap();
            interface.validate().unwrap();
        }
        assert!(interface("kernel_unknown").is_err());
    }
    #[test]
    fn corrupt_output_guards_are_rejected() {
        let size = logic::self_test::SELF_TEST_NUM_CHECKS * 4;
        let buffer = guarded(size);
        assert_eq!(
            output(&buffer).unwrap(),
            vec![super::super::SENTINEL; size / 4]
        );
        for index in [0, GUARD - 1, GUARD + size, GUARD * 2 + size - 1] {
            let mut buffer = guarded(size);
            buffer.bytes[index] = 0;
            assert!(output(&buffer).is_err());
        }
    }
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "metal-self-test-manifest-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn group_fixture(root: &Path) -> serde_json::Value {
        use serde_json::json;
        let kernel = "kernel_self_test_ethereum";
        let mut cases = Vec::new();
        for case in logic::self_test::metadata::CASES
            .iter()
            .filter(|case| case.kernel == kernel)
        {
            let relative = format!("cases/{}", case.name);
            let directory = root.join(&relative);
            fs::create_dir_all(&directory).unwrap();
            let manifest = json!({"schema": 1, "self_test": {"kernel": kernel, "name": case.name, "slot": case.slot, "entry": case.metal_entry}, "source_revision": "test", "source_sha256": {"logic.rs": "test"}, "compiler_sha256": "compiler", "rustc": "test"});
            let bytes = serde_json::to_vec(&manifest).unwrap();
            fs::write(directory.join("kernel.build.json"), &bytes).unwrap();
            cases.push(json!({"name": case.name, "slot": case.slot, "entry": case.metal_entry, "directory": relative, "manifest_sha256": hex::encode(Sha256::digest(&bytes))}));
        }
        json!({"schema": 1, "kind": "self-test-group", "kernel": kernel, "cases": cases, "source_revision": "test", "source_sha256": {"logic.rs": "test"}, "compiler_sha256": "compiler", "rustc": "test"})
    }
    #[test]
    fn incomplete_misidentified_and_mixed_provenance_groups_are_rejected() {
        let fixture = Fixture::new();
        let complete = group_fixture(&fixture.0);
        let check = |group: &serde_json::Value| {
            fs::write(
                fixture.0.join("self-tests.json"),
                serde_json::to_vec(group).unwrap(),
            )
            .unwrap();
            validate_group(&fixture.0, "kernel_self_test_ethereum")
        };
        check(&complete).unwrap();
        let mut missing = complete.clone();
        missing["cases"].as_array_mut().unwrap().pop();
        assert!(check(&missing).unwrap_err().contains("incomplete"));
        for (field, value) in [
            ("name", serde_json::json!("wrong.check")),
            ("slot", serde_json::json!(999)),
            ("entry", serde_json::json!("wrong_entry")),
            ("directory", serde_json::json!("../escape")),
            ("manifest_sha256", serde_json::json!("bad hash")),
        ] {
            let mut changed = complete.clone();
            changed["cases"][0][field] = value;
            assert!(check(&changed).is_err(), "{field}");
        }
        let mut mixed = complete.clone();
        mixed["compiler_sha256"] = "another compiler".into();
        assert!(check(&mixed).unwrap_err().contains("provenance"));
        let relative = complete["cases"][0]["directory"].as_str().unwrap();
        fs::remove_file(fixture.0.join(relative).join("kernel.build.json")).unwrap();
        assert!(check(&complete).is_err());
    }
}
