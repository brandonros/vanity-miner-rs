#![cfg(all(feature = "metal", feature = "shallenge", target_os = "macos"))]
use logic::search::xoroshiro::BatchSeed;
use std::path::PathBuf;
use vanity_miner::modes::shallenge::metal::ShallengeTransport;

fn artifacts() -> PathBuf {
    super::support::artifacts("VANITY_METAL_ARTIFACTS", "shallenge")
}

#[test]
#[ignore = "build scripts/build-metal.sh --mode shallenge first; requires Apple GPU"]
fn application_batches_compare_every_lane_and_preserve_guards() {
    let mut engine = ShallengeTransport::load(&artifacts(), 257, 64, true).unwrap();
    for count in [1, 31, 32, 33, 64, 65, 257] {
        for (seed, width, start) in [(12345, 32, 0), (u64::MAX, 33, 31), (0, 1, u64::MAX - 256)] {
            for (username, target) in [
                ("a", [0; 32]),
                ("brandonros", [255; 32]),
                ("abcdefghijklmnopqrstuvwxyz1234", [128; 32]),
            ] {
                engine
                    .evaluate(
                        &BatchSeed { seed, width },
                        &target,
                        username.as_bytes(),
                        start,
                        count,
                    )
                    .unwrap();
            }
        }
    }
    // Valid launch memory but invalid application requests must preserve ERROR,
    // never turn an error into a miss or let a winning lane conceal it.
    for (width, username) in [
        (0, "a"),
        (u64::from(u32::MAX) + 1, "a"),
        (32, ""),
        (32, "abcdefghijklmnopqrstuvwxyz12345"),
    ] {
        assert_eq!(
            engine
                .evaluate(
                    &BatchSeed { seed: 0, width },
                    &[255; 32],
                    username.as_bytes(),
                    0,
                    33
                )
                .err()
                .as_deref(),
            Some("device candidate evaluation failed")
        );
    }
    let before = engine.launches;
    assert!(
        engine
            .evaluate(
                &BatchSeed { seed: 0, width: 32 },
                &[255; 32],
                b"a",
                u64::MAX,
                2
            )
            .is_err()
    );
    assert!(
        engine
            .evaluate(&BatchSeed { seed: 0, width: 32 }, &[255; 32], b"a", 0, 0)
            .is_err()
    );
    assert_eq!(engine.launches, before);
}

#[test]
#[ignore = "build scripts/build-metal.sh --mode shallenge first; requires Apple GPU"]
fn bounded_cli_search_verifies_winners_and_finishes_without_matches() {
    for target in ["ff".repeat(32), "00".repeat(32)] {
        let output = super::support::command(&artifacts())
            .args([
                "--batches",
                "4",
                "--batch-size",
                "33",
                "--seed",
                "12345",
                "--verify",
                "shallenge",
                "--username",
                "brandonros",
                "--target-hash",
                &target,
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("132 total nonces"), "{stdout}");
        assert_eq!(
            stdout.contains("[shallenge] nonce="),
            target.starts_with("ff")
        );
    }
}

#[test]
#[ignore = "build scripts/build-metal.sh --mode shallenge first; requires Apple GPU"]
fn large_partial_grid_checks_every_application_result() {
    let mut engine = ShallengeTransport::load(&artifacts(), 65537, 64, true).unwrap();
    for target in [[128; 32], [0; 32]] {
        engine
            .evaluate(
                &BatchSeed {
                    seed: u64::MAX,
                    width: 4096,
                },
                &target,
                b"brandonros",
                u64::from(u32::MAX) + 31,
                65537,
            )
            .unwrap();
    }
}

#[test]
fn mismatched_artifact_hash_and_bindings_are_rejected_before_loading() {
    use sha2::{Digest, Sha256};
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../target/metal/refusal-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let library = b"not a library; refusal must happen before Metal load";
    let bindings = br#"{"entry":"wrong","dispatch":"grid1d","buffers":[]}"#;
    std::fs::write(directory.join("kernel.metallib"), library).unwrap();
    std::fs::write(directory.join("kernel.bindings.json"), bindings).unwrap();
    let mut manifest = serde_json::json!({"schema":1,"artifacts":{"kernel.metallib":"incorrect","kernel.bindings.json":hex::encode(Sha256::digest(bindings))}});
    std::fs::write(directory.join("kernel.build.json"), manifest.to_string()).unwrap();
    let error = ShallengeTransport::load(&directory, 1, 1, false)
        .err()
        .unwrap();
    assert!(error.contains("hash mismatch"), "{error}");
    manifest["artifacts"]["kernel.metallib"] = hex::encode(Sha256::digest(library)).into();
    std::fs::write(directory.join("kernel.build.json"), manifest.to_string()).unwrap();
    let error = ShallengeTransport::load(&directory, 1, 1, false)
        .err()
        .unwrap();
    assert!(error.contains("application ABI"), "{error}");
    let legacy = r#"{"entry":"kernel_shallenge","dispatch":"grid1d","buffers":[{"argument":0,"index":0,"minimum_bytes":104,"alignment":8,"access":"read"},{"argument":1,"index":1,"minimum_bytes":272,"alignment":4,"access":"read_write"},{"argument":2,"index":2,"minimum_bytes":260,"alignment":4,"access":"write"}]}"#;
    std::fs::write(directory.join("kernel.bindings.json"), legacy).unwrap();
    manifest["artifacts"]["kernel.bindings.json"] = hex::encode(Sha256::digest(legacy)).into();
    std::fs::write(directory.join("kernel.build.json"), manifest.to_string()).unwrap();
    let error = ShallengeTransport::load(&directory, 1, 1, false)
        .err()
        .unwrap();
    assert!(error.contains("application ABI"), "{error}");
    std::fs::remove_dir_all(directory).unwrap();
}
