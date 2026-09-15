use super::*;
use rand::RngCore;
use rsa::pkcs8::{EncodePublicKey, LineEnding};
use rsa::{
    pkcs8::{DecodePrivateKey, DecodePublicKey},
    traits::PrivateKeyParts,
};
use std::path::PathBuf;

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn constructive_keys_round_trip_and_pass_libressl_checks() {
    check_runner();
}

fn check_runner() {
    let mut random = [0; 16];
    OsRng.fill_bytes(&mut random);
    let dir = Directory(
        std::env::temp_dir().join(format!("vanity-modulus-test-{}", hex::encode(random))),
    );
    std::fs::create_dir(&dir.0).unwrap();
    for (i, (prefix, suffix)) in [("abc", ""), ("", "fed"), ("d", "b")]
        .into_iter()
        .enumerate()
    {
        let private_out = dir.0.join(format!("private-{i}.pem"));
        let public_out = dir.0.join(format!("public-{i}.pem"));
        let config = ModulusSearch {
            prefix: prefix.into(),
            suffix: suffix.into(),

            workers: 2,
        };
        let control = Arc::new(SearchControl::new());
        let report = run_cpu(&config, control).unwrap();
        assert!(report.found);
        assert!(report.q_candidates_tested > 0);
        let record = report.output.as_ref().unwrap();
        let key = RsaPrivateKey::from_pkcs8_der(&crate::modes::tests::console_field(
            record,
            "private_key_pkcs8",
        ))
        .unwrap();
        assert_eq!(
            crate::modes::tests::console_field(record, "public_key"),
            key.n().to_bytes_be()
        );
        std::fs::write(
            &private_out,
            key.to_pkcs8_pem(LineEnding::LF).unwrap().as_bytes(),
        )
        .unwrap();
        std::fs::write(
            &public_out,
            key.to_public_key()
                .to_public_key_pem(LineEnding::LF)
                .unwrap(),
        )
        .unwrap();
        key.validate().unwrap();
        assert_eq!(key.n().bits(), 2048);
        assert!(key.primes().iter().all(|prime| prime.bits() == 1024));
        assert!(sufficiently_separated(&key.primes()[0], &key.primes()[1]));
        assert!(config.validate().is_ok());
        let constraints = ModulusConstraints::new(prefix, suffix).unwrap();
        assert!(constraints.pattern.matches(&key.n().to_bytes_be()));
        let public_pem = std::fs::read_to_string(&public_out).unwrap();
        let public = rsa::RsaPublicKey::from_public_key_pem(&public_pem).unwrap();
        assert!(public == key.to_public_key());
        if std::process::Command::new("openssl")
            .arg("version")
            .output()
            .is_ok()
        {
            let result = std::process::Command::new("openssl")
                .args(["rsa", "-in"])
                .arg(&private_out)
                .args(["-check", "-noout"])
                .output()
                .unwrap();
            assert!(result.status.success());
        }
    }
}

#[test]
fn impossible_patterns_and_entropy_constraints() {
    for (prefix, suffix) in [("7", ""), ("", "e"), ("0x8", ""), ("g", "")] {
        assert!(ModulusConstraints::new(prefix, suffix).is_err());
    }
    assert!(ModulusConstraints::new(&"f".repeat(192), "").is_err());
}

#[test]
fn long_prefix_allows_single_candidate_interval() {
    let one = BigUint::from(1u8);
    let p = (&one << 1024usize) - BigUint::from(109u8);
    let q = (&one << 1023usize) + BigUint::from(123u8);
    let n = &p * &q;
    let encoded = hex::encode(n.to_bytes_be());
    let constraints = ModulusConstraints::new(&encoded[..256], "").unwrap();
    let progression = constraints.progression(&p).unwrap();
    assert_eq!(*progression.count, one);
    assert_eq!(*progression.first, q);
    assert_eq!(progression.search_budget(), 1);
    assert!(constraints.pattern.matches(&n.to_bytes_be()));
    assert!(ModulusConstraints::new(&encoded[..258], "").is_ok());
    assert!(ModulusConstraints::new(&encoded[..254], "ab").is_ok());
    assert!(ModulusConstraints::new(&encoded[..256], "ab").is_ok());
}

#[test]
fn small_intervals_have_bounded_nonempty_cpu_batches() {
    for size in [1u64, 2, 63, 64, 65, 65535, 65536, 65537] {
        let progression = QProgression {
            first: Zeroizing::new(BigUint::from(1u8)),
            stride: BigUint::from(2u8),
            count: Zeroizing::new(BigUint::from(size)),
        };
        let budget = progression.search_budget();
        assert_eq!(budget, size.min(65536));
        let start_choices = &*progression.count - BigUint::from(budget - 1);
        assert!(start_choices >= BigUint::from(1u8));
        let batches: Vec<_> = (0..budget)
            .step_by(64)
            .map(|start| (budget - start).min(64))
            .collect();
        assert!(batches.iter().all(|count| (1..=64).contains(count)));
        assert_eq!(batches.iter().sum::<u64>(), budget);
    }
}

#[test]
fn rejects_patterns_above_factor_separation_bound() {
    for length in [25, 26, 100] {
        for suffix in ["", "1", "abcd"] {
            let error = ModulusConstraints::new(&"f".repeat(length), suffix)
                .err()
                .expect("impossible all-f prefix must be rejected");
            assert!(error.contains("factor separation"));
        }
    }
    for suffix in ["", "1", "abcd"] {
        assert!(ModulusConstraints::new(&"f".repeat(24), suffix).is_ok());
        // Length alone is not the limit: a nearby lower interval is valid.
        assert!(ModulusConstraints::new(&format!("{}e", "f".repeat(24)), suffix).is_ok());
        assert!(ModulusConstraints::new("a3b6", suffix).is_ok());
    }
}

#[test]
fn interval_and_residue_match_all_sampled_candidates() {
    for (prefix, suffix) in [("abc", ""), ("", "fed"), ("d", "b"), ("fffffff", "1")] {
        let constraints = ModulusConstraints::new(prefix, suffix).unwrap();
        let (p, progression) = loop {
            let p = constraints.random_p_candidate();
            if let Some(progression) = constraints.progression(&p) {
                break (p, progression);
            }
        };
        for index in [
            BigUint::from(0u8),
            &*progression.count - BigUint::from(1u8),
            OsRng.gen_biguint_below(&progression.count),
        ] {
            let q = &*progression.first + index * &progression.stride;
            assert_eq!(q.bits(), 1024);
            let n = &p * q;
            assert_eq!(n.bits(), 2048);
            assert!(
                constraints
                    .pattern
                    .matches(&fixed_bytes::<256>(&n).unwrap()[..])
            );
        }
    }
}
