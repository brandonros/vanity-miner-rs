//! Process-level CPU smoke tests for global flag propagation and clean exit.
#![cfg(all(
    not(feature = "metal"),
    feature = "shallenge",
    feature = "ethereum",
    feature = "bitcoin",
    feature = "solana",
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-pss"
))]
use p256::pkcs8::{EncodePrivateKey, LineEnding};
use std::{
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[path = "support/rsa_factors.rs"]
mod factors;

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn check(args: &[&str], record_marker: &str) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vanity-miner"))
        .arg("--exit-on-first-match")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "first-match CLI did not stop: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.matches(record_marker).count(), 1, "{stdout}");
    assert!(stdout.contains("| 1 matches in"), "{stdout}");
}

#[test]
fn cpu_cli_prints_one_record_and_exits_successfully() {
    let dir = Directory(std::env::temp_dir().join(format!(
        "vanity-first-match-{}-{}",
        std::process::id(),
        rand::random::<u64>()
    )));
    std::fs::create_dir(&dir.0).unwrap();
    let message = dir.0.join("message.bin");
    std::fs::write(&message, b"first match").unwrap();
    let p256 = dir.0.join("p256.pem");
    let key = p256::SecretKey::from_slice(&[1; 32]).unwrap();
    std::fs::write(&p256, key.to_pkcs8_pem(LineEnding::LF).unwrap().as_bytes()).unwrap();
    let rsa = dir.0.join("rsa.pem");
    let key = rsa::RsaPrivateKey::from_p_q(
        rsa::BigUint::from_bytes_be(&factors::P),
        rsa::BigUint::from_bytes_be(&factors::Q),
        rsa::BigUint::from(65537u32),
    )
    .unwrap();
    std::fs::write(&rsa, key.to_pkcs8_pem(LineEnding::LF).unwrap().as_bytes()).unwrap();
    for command in ["ethereum-vanity", "bitcoin-vanity", "solana-vanity"] {
        check(&[command], "Vanity match: rng_seed =");
    }
    check(
        &[
            "shallenge",
            "--username",
            "miner",
            "--target-hash",
            &"ff".repeat(32),
        ],
        "NEW GLOBAL BEST hash:",
    );
    check(
        &["p256-public-key-vanity", "--threads", "2"],
        "[p256-public-key] private_key=",
    );
    check(
        &[
            "p256-signature-vanity",
            "--threads",
            "2",
            "--key",
            p256.to_str().unwrap(),
            "--message",
            message.to_str().unwrap(),
            "--search-source",
            "ephemeral",
        ],
        "[p256-signature] signature=",
    );
    check(
        &[
            "rsa-pss-signature-vanity",
            "--threads",
            "2",
            "--key",
            rsa.to_str().unwrap(),
            "--message",
            message.to_str().unwrap(),
            "--salt-length",
            "0",
        ],
        "[rsa-pss] signature=",
    );
    // RSA modulus generation is stochastic and already has a bounded known-pair
    // worker test; flag parsing and the shared stop/restart policy cover it.
}
