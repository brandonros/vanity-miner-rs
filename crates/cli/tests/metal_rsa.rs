#![cfg(all(feature = "metal", feature = "rsa-modulus", target_os = "macos"))]
#[path = "support/rsa_factors.rs"]
mod factors;
use logic::{modes::rsa_modulus::SearchConfig, search::hex_pattern::HexPattern};
use num_bigint_dig::BigUint;
use std::{path::PathBuf, process::Command};
use vanity_miner::{modes::rsa_modulus::pipeline::verify_pair, runner::metal::rsa::RsaTransport};
fn artifacts() -> PathBuf {
    std::env::var_os("VANITY_METAL_RSA_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/metal/rsa-modulus")
        })
}
fn fixed<const N: usize>(n: &BigUint) -> [u8; N] {
    let raw = n.to_bytes_be();
    let mut out = [0; N];
    out[N - raw.len()..].copy_from_slice(&raw);
    out
}
fn config() -> SearchConfig {
    let n = BigUint::from_bytes_be(&factors::P) * BigUint::from_bytes_be(&factors::Q);
    let mut suffix = [0; 256];
    suffix[255] = 1;
    SearchConfig {
        lower: fixed(&n),
        upper: fixed(&n),
        p_min: factors::P,
        p_count: fixed(&BigUint::from(1u8)),
        suffix,
        seed: [42; 32],
        worker: 7,
        suffix_bits: 1,
        reserved: 0,
    }
}
#[test]
#[ignore = "build scripts/build-metal.py --mode rsa-modulus first; requires Apple GPU"]
fn independent_candidates_and_host_key_verification() {
    use logic::modes::rsa_modulus::Pair;
    use rsa::{RsaPrivateKey, pkcs8::DecodePrivateKey, traits::PublicKeyParts};
    let c = config();
    let pattern = HexPattern::new("", "", 256).unwrap();
    for (capacity, group, audit) in [(1, 1, true), (3, 4, true), (3, 4, false)] {
        let mut engine = RsaTransport::load(&artifacts(), capacity, group, audit).unwrap();
        for start in [0, 101, 0] {
            let batch = engine.evaluate(&c, &pattern, &[], start, capacity).unwrap();
            assert_eq!(batch.matches, capacity);
            let (lane, winner) = batch.winner(capacity).unwrap().unwrap();
            assert_eq!(winner.bytes[..128], factors::P);
            assert_eq!(winner.bytes[128..], factors::Q);
            let pair = Pair {
                p: factors::P,
                q: factors::Q,
                id: start + u64::from(lane),
            };
            let output = zeroize::Zeroizing::new(verify_pair(&c, &pattern, &pair).unwrap());
            let der = zeroize::Zeroizing::new(
                hex::decode(
                    output
                        .lines()
                        .find_map(|line| line.strip_prefix("[rsa-modulus] private_key_pkcs8="))
                        .unwrap(),
                )
                .unwrap(),
            );
            let key = RsaPrivateKey::from_pkcs8_der(&der).unwrap();
            key.validate().unwrap();
            assert_eq!(fixed::<256>(key.n()), c.lower);
        }
        // A different request is valid: no saved task may affect the next launch.
        let mut changed = c;
        changed.seed[0] ^= 1;
        assert_eq!(
            engine
                .evaluate(&changed, &pattern, &[], 50, capacity)
                .unwrap()
                .matches,
            capacity
        );
        // Exercise the q sampler on the GPU, not only a single-value range.
        let mut range = c;
        range.upper = fixed(
            &(BigUint::from_bytes_be(&factors::P)
                * (BigUint::from_bytes_be(&factors::Q) + BigUint::from(8u8))),
        );
        assert!(
            engine
                .evaluate(&range, &pattern, &[], 9, capacity)
                .unwrap()
                .matches
                >= 1
        );
        let launches = engine.launches;
        for (start, count) in [(0, 0), (0, capacity + 1), (u64::MAX, 2)] {
            assert!(engine.evaluate(&c, &pattern, &[], start, count).is_err());
        }
        assert_eq!(engine.launches, launches);
        let miss = HexPattern::new("00", "", 256).unwrap();
        assert!(
            engine
                .evaluate(&c, &miss, &[], 0, capacity)
                .unwrap()
                .winner(capacity)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            engine
                .evaluate(&c, &pattern, &[], u64::MAX, 1)
                .unwrap()
                .matches,
            1
        );
        let mut invalid = c;
        invalid.p_count = [0; 128];
        assert!(
            engine
                .evaluate(&invalid, &pattern, &[], 0, capacity)
                .is_err()
        );
    }
}
#[test]
fn invalid_dispatch_is_rejected_before_artifact_loading() {
    for (capacity, group) in [(0, 1), (1, 0), (1, 1025), (1_048_577, 1)] {
        assert!(RsaTransport::load(&PathBuf::from("missing"), capacity, group, false).is_err());
    }
}
#[test]
#[ignore = "requires freshly built RSA artifacts and Apple GPU"]
fn bounded_cli_search_and_entropy_policy() {
    let binary = env!("CARGO_BIN_EXE_vanity-miner");
    let run = |extra: &[&str]| {
        Command::new(binary)
            .args([
                "--verify",
                "--batch-size",
                "1",
                "--threads-per-group",
                "1",
                "--batches",
                "2",
                "--metal-artifacts",
            ])
            .arg(artifacts())
            .args(["rsa-modulus-vanity"])
            .args(extra)
            .output()
            .unwrap()
    };
    let out = run(&[]);
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("Metal RSA: 2 launches"));
    let out = run(&["--steps-per-launch", "1"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unexpected argument"));
    let out = run(&["--seed", "1"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("OS cryptographic entropy"));
}
