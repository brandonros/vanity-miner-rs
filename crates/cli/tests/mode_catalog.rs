//! Every production mode has its kernel crates, CLI adapters, Cargo features,
//! CI matrix entry and smoke command. Adding a mode means touching all of them.
use std::{collections::BTreeSet, fs, path::PathBuf};

const MODES: [&str; 8] = [
    "shallenge",
    "bitcoin",
    "ethereum",
    "solana",
    "p256-public-key",
    "p256-signature",
    "rsa-modulus",
    "rsa-pss",
];

#[test]
fn every_mode_is_wired_everywhere() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let read = |path: &str| fs::read_to_string(root.join(path)).expect(path);
    let modes: BTreeSet<_> = MODES.iter().map(|mode| mode.to_string()).collect();
    assert_eq!(modes.len(), MODES.len(), "duplicate mode");

    let kernels: BTreeSet<_> = fs::read_dir(root.join("crates/kernels"))
        .unwrap()
        .flatten()
        .filter(|entry| entry.path().join("Cargo.toml").is_file())
        .map(|entry| entry.file_name().into_string().unwrap())
        .filter(|name| !name.starts_with("self-test-"))
        .collect();
    assert_eq!(kernels, modes, "production kernel crates differ from the catalog");

    let workflow = read(".github/workflows/metal.yaml");
    let matrix = workflow
        .lines()
        .find_map(|line| line.trim().strip_prefix("mode: ["))
        .and_then(|line| line.strip_suffix(']'))
        .expect("CI kernel matrix");
    let matrix: BTreeSet<_> = matrix.split(',').map(|mode| mode.trim().to_string()).collect();
    assert_eq!(matrix, modes, "CI kernel matrix differs from the catalog");

    let manifest = read("crates/cli/Cargo.toml");
    let features = manifest.split("[features]").nth(1).expect("features table");
    let features = features.split("\n[").next().unwrap();
    let smoke = read("scripts/smoke-metal.sh");
    for mode in MODES {
        let module = mode.replace('-', "_");
        for path in [
            format!("crates/kernels/{mode}/src/contract.rs"),
            format!("crates/kernels/self-test-{mode}/Cargo.toml"),
            format!("crates/cli/src/modes/{module}/cpu.rs"),
            format!("crates/cli/src/modes/{module}/metal.rs"),
            format!("crates/cli/src/modes/{module}/device.rs"),
            format!("crates/cli/tests/metal/{module}.rs"),
        ] {
            assert!(root.join(&path).is_file(), "missing mode component: {path}");
        }
        for feature in [mode.to_string(), format!("self_test_{module}")] {
            assert!(
                features.lines().any(|line| line.starts_with(&format!("{feature} ="))),
                "missing mode feature: {feature}"
            );
        }
        assert!(
            smoke.lines().any(|line| line.starts_with(&format!("run {mode} "))),
            "missing smoke command: {mode}"
        );
    }
}
