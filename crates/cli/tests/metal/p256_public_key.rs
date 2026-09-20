#![cfg(all(feature = "metal", feature = "p256-public-key", target_os = "macos"))]
use logic::{
    crypto::p256::PublicTarget,
    modes::p256_public_key::{P256PublicRequest, p256_public},
    search::hex_pattern::HexPattern,
};
use std::path::PathBuf;
use vanity_miner::modes::p256_public_key::metal::P256PublicTransport;

fn artifacts() -> PathBuf {
    super::support::artifacts("VANITY_METAL_P256_PUBLIC_ARTIFACTS", "p256-public-key")
}
fn request(target: u32) -> P256PublicRequest {
    // Public test seed only; production CLI always obtains OS entropy.
    P256PublicRequest {
        seed: [0x42; 32],
        worker: 7,
        target,
        reserved: 0,
    }
}

#[test]
fn invalid_dispatch_is_rejected_before_loading_artifacts() {
    for (capacity, group) in [(0, 1), (1_048_577, 1), (1, 0), (1, 1025)] {
        assert!(
            P256PublicTransport::load(&PathBuf::from("missing"), capacity, group, true)
                .err()
                .unwrap()
                .contains("dispatch")
        );
    }
}

#[test]
#[ignore = "build scripts/build-metal.sh --mode p256-public-key first; requires Apple GPU"]
fn targets_multilane_winners_misses_and_errors_match_cpu() {
    for audit in [true, false] {
        let mut engine = P256PublicTransport::load(&artifacts(), 5, 4, audit).unwrap();
        for (target, representation) in [
            PublicTarget::X,
            PublicTarget::Y,
            PublicTarget::Xy,
            PublicTarget::Uncompressed,
        ]
        .into_iter()
        .enumerate()
        {
            let request = request(target as u32);
            let all = HexPattern::new("", "", representation.width()).unwrap();
            let expected = p256_public(&request, 9, &all);
            let point: [u8; 65] = expected.bytes[..65].try_into().unwrap();
            let encoded = hex::encode(representation.bytes(&point));
            let exact = HexPattern::new(&encoded, &encoded, representation.width()).unwrap();
            let winner = engine.evaluate(&request, &exact, &[], 7, 5).unwrap();
            assert_eq!((winner.matches, winner.lane), (1, 2));
            assert_eq!(winner.candidate.bytes, expected.bytes);
            assert_eq!(
                engine
                    .evaluate(&request, &exact, &[], 12, 5)
                    .unwrap()
                    .matches,
                0
            );
            assert_eq!(
                engine.evaluate(&request, &all, &[], 0, 5).unwrap().matches,
                5
            );
            assert_eq!(
                engine
                    .evaluate(&request, &all, &[], u64::MAX, 1)
                    .unwrap()
                    .matches,
                1
            );
            let mut wrong = representation.bytes(&point).to_vec();
            wrong[0] ^= 0x80;
            let miss = HexPattern::new(&hex::encode(wrong), "", representation.width()).unwrap();
            assert_eq!(
                engine.evaluate(&request, &miss, &[], 9, 1).unwrap().matches,
                0
            );
        }
        let all = HexPattern::new("", "", 32).unwrap();
        assert_eq!(
            engine
                .evaluate(&request(4), &all, &[], 0, 3)
                .err()
                .as_deref(),
            Some("device candidate evaluation failed")
        );
        let launches = engine.launches;
        for (start, count) in [(0, 0), (0, 6), (u64::MAX, 2)] {
            assert!(
                engine
                    .evaluate(&request(0), &all, &[], start, count)
                    .is_err()
            );
        }
        assert_eq!(engine.launches, launches);
    }
    let mut scalar = P256PublicTransport::load(&artifacts(), 1, 1, true).unwrap();
    assert_eq!(
        scalar
            .evaluate(
                &request(0),
                &HexPattern::new("", "", 32).unwrap(),
                &[],
                0,
                1
            )
            .unwrap()
            .matches,
        1
    );
}

#[test]
#[ignore = "build scripts/build-metal.sh --mode p256-public-key first; requires Apple GPU"]
fn bounded_cli_exports_verified_keys_for_every_target() {
    use p256::elliptic_curve::sec1::ToEncodedPoint;
    for target in ["x", "y", "xy", "uncompressed"] {
        let result = super::support::command(&artifacts())
            .args([
                "--batches",
                "2",
                "--batch-size",
                "1",
                "--threads-per-group",
                "1",
                "--verify",
                "p256-public-key-vanity",
                "--target",
                target,
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let stdout = String::from_utf8(result.stdout).unwrap();
        assert!(stdout.contains("2 total keys"), "{stdout}");
        let values = |field: &str| -> Vec<Vec<u8>> {
            stdout
                .lines()
                .filter_map(|line| line.strip_prefix(&format!("[p256-public-key] {field}=")))
                .map(|value| hex::decode(value).unwrap())
                .collect()
        };
        let private = values("private_key");
        let public = values("sec1_public_key");
        assert_eq!(private.len(), 2);
        assert_eq!(public.len(), 2);
        for (private, public) in private.iter().zip(&public) {
            let key = p256::SecretKey::from_slice(private).unwrap();
            assert_eq!(key.public_key().to_encoded_point(false).as_bytes(), public);
        }
    }
    let rejected = super::support::command(&artifacts())
        .args(["--seed", "1", "--batches", "1", "p256-public-key-vanity"])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("--seed is not supported"));
}
