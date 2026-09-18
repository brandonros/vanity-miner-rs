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
fn prepare_resume_winner_retirement_and_host_key_verification() {
    use rsa::{RsaPrivateKey, pkcs8::DecodePrivateKey, traits::PublicKeyParts};
    let c = config();
    let pattern = HexPattern::new("", "", 256).unwrap();
    for (capacity, group, audit) in [(1, 1, true), (3, 4, true), (1, 1, false)] {
        let mut engine =
            RsaTransport::load(&artifacts(), &c, &pattern, capacity, 1, group, audit).unwrap();
        let (prepared, pairs) = engine.cycle(&c, &pattern, 0).unwrap();
        assert!(pairs.is_empty());
        assert_eq!(prepared.p_tested, capacity);
        assert_eq!(prepared.ranges, capacity);
        assert_eq!(prepared.q_tested, 0);
        let (searched, pairs) = engine.cycle(&c, &pattern, u64::from(capacity)).unwrap();
        assert_eq!(searched.p_tested, 0);
        assert_eq!(searched.q_tested, capacity);
        assert_eq!(searched.matches, capacity);
        for (lane, pair) in pairs.iter().enumerate() {
            assert_eq!(pair.id, lane as u64);
            assert_eq!(pair.p, factors::P);
            assert_eq!(pair.q, factors::Q);
        }
        let output = zeroize::Zeroizing::new(verify_pair(&c, &pattern, &pairs[0]).unwrap());
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
        let (restarted, pairs) = engine.cycle(&c, &pattern, u64::from(capacity) * 2).unwrap();
        assert_eq!(restarted.p_tested, capacity);
        assert_eq!(restarted.q_tested, 0);
        assert!(pairs.is_empty());
        let (_, pairs) = engine.cycle(&c, &pattern, u64::from(capacity) * 3).unwrap();
        assert_eq!(pairs[0].id, u64::from(capacity) * 2);
        let launches = engine.launches;
        let mut changed = c;
        changed.seed[0] ^= 1;
        assert!(engine.cycle(&changed, &pattern, 50).is_err());
        if capacity > 1 {
            assert!(engine.cycle(&c, &pattern, u64::MAX).is_err());
        }
        assert_eq!(engine.launches, launches);
    }
    let miss = HexPattern::new("00", "", 256).unwrap();
    let mut engine = RsaTransport::load(&artifacts(), &c, &miss, 1, 1, 1, true).unwrap();
    engine.cycle(&c, &miss, 0).unwrap();
    let (counts, pairs) = engine.cycle(&c, &miss, 1).unwrap();
    assert_eq!(counts.q_tested, 1);
    assert_eq!(counts.matches, 0);
    assert!(pairs.is_empty());
    assert_eq!(engine.cycle(&c, &miss, 2).unwrap().0.p_tested, 1);
}
#[test]
fn invalid_dispatch_is_rejected_before_artifact_loading() {
    let c = config();
    let p = HexPattern::new("", "", 256).unwrap();
    for (capacity, steps, group) in [
        (0, 1, 1),
        (1, 0, 1),
        (1, 1025, 1),
        (1, 1, 0),
        (1, 1, 1025),
        (1_048_577, 1, 1),
    ] {
        assert!(
            RsaTransport::load(
                &PathBuf::from("missing"),
                &c,
                &p,
                capacity,
                steps,
                group,
                false
            )
            .is_err()
        );
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
            .args(["rsa-modulus-vanity", "--steps-per-launch", "1"])
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
    let out = run(&["--seed", "1"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("OS cryptographic entropy"));
}
