#![cfg(all(feature = "metal", feature = "rsa-pss", target_os = "macos"))]
#[path = "support/rsa_factors.rs"]
mod factors;
use logic::{modes::rsa_pss::RsaPssRequest, search::hex_pattern::HexPattern};
use rsa::{
    BigUint, Pss, RsaPrivateKey,
    pkcs8::{EncodePrivateKey, LineEnding},
    traits::PrivateKeyParts,
};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, process::Command};
use vanity_miner::runner::metal::rsa_pss::RsaPssTransport;

fn artifacts() -> PathBuf {
    std::env::var_os("VANITY_METAL_RSA_PSS_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/metal/rsa-pss")
        })
}
fn key() -> RsaPrivateKey {
    let mut key = RsaPrivateKey::from_primes(
        vec![
            BigUint::from_bytes_be(&factors::P),
            BigUint::from_bytes_be(&factors::Q),
        ],
        BigUint::from(65537u32),
    )
    .unwrap();
    key.precompute().unwrap();
    key.validate().unwrap();
    key
}
fn fixed<const N: usize>(value: &BigUint) -> [u8; N] {
    let bytes = value.to_bytes_be();
    let mut out = [0; N];
    out[N - bytes.len()..].copy_from_slice(&bytes);
    out
}
fn request(key: &RsaPrivateKey, message: &[u8], salt_length: u32) -> RsaPssRequest {
    RsaPssRequest {
        p: fixed(&key.primes()[0]),
        q: fixed(&key.primes()[1]),
        dp: fixed(key.dp().unwrap()),
        dq: fixed(key.dq().unwrap()),
        q_inv: fixed(&key.crt_coefficient().unwrap()),
        digest: Sha256::digest(message).into(),
        salt: [0x42; 222],
        reserved: [0; 2],
        offset: 0,
        length: 0,
        source: 0,
        salt_length,
    }
}
fn verify(
    key: &RsaPrivateKey,
    request: &RsaPssRequest,
    message: &[u8],
    counter: u64,
    signature: &[u8],
) {
    let mut message = message.to_vec();
    let mut salt = request.salt[..request.salt_length as usize].to_vec();
    if request.source == 0 {
        logic::search::salt_counter::write_salt_counter(
            &request.salt[..salt.len()],
            counter,
            &mut salt,
        )
        .unwrap();
    } else {
        logic::search::message_window::write_message_counter(
            &mut message,
            request.offset as usize,
            request.length as usize,
            u128::from(counter),
        )
        .unwrap();
    }
    key.to_public_key()
        .verify(
            Pss::new_with_salt::<Sha256>(salt.len()),
            &Sha256::digest(&message),
            signature,
        )
        .unwrap();
}

#[test]
fn invalid_dispatch_rejected_before_artifact_loading() {
    for (count, group) in [(0, 1), (1_048_577, 1), (1, 0), (1, 1025)] {
        assert!(RsaPssTransport::load(&PathBuf::from("missing"), count, group, true).is_err());
    }
}

#[test]
#[ignore = "build scripts/build-metal.py --mode rsa-pss first; requires Apple GPU"]
fn salt_and_message_candidates_crt_validation_and_winners() {
    let key = key();
    let any = HexPattern::new("", "", 256).unwrap();
    let message: Vec<u8> = (0..130).map(|n| n as u8).collect();
    let mut engine = RsaPssTransport::load(&artifacts(), 3, 4, true).unwrap();
    for (salt_length, start, count) in [(0, 0, 1), (1, 254, 2), (32, 0, 3), (222, u64::MAX, 1)] {
        eprintln!("RSA-PSS salt length={salt_length} start={start} count={count}");
        let r = request(&key, &message, salt_length);
        let batch = engine.evaluate(&r, &any, &message, start, count).unwrap();
        assert_eq!(batch.matches, count);
        verify(
            &key,
            &r,
            &message,
            start + u64::from(batch.lane),
            &batch.candidate.bytes,
        );
    }
    for (length, offset, window) in [
        (1, 0, 1),
        (55, 54, 1),
        (56, 0, 8),
        (64, 56, 8),
        (130, 55, 9),
    ] {
        eprintln!("RSA-PSS message length={length} offset={offset} window={window}");
        let input = &message[..length];
        let mut r = request(&key, input, 32);
        r.source = 1;
        r.offset = offset;
        r.length = window;
        let batch = engine.evaluate(&r, &any, input, 1, 2).unwrap();
        assert_eq!(batch.matches, 2);
        verify(
            &key,
            &r,
            input,
            1 + u64::from(batch.lane),
            &batch.candidate.bytes,
        );
    }
    let r = request(&key, &message, 32);
    let batch = engine.evaluate(&r, &any, &message, 0, 1).unwrap();
    let signature = hex::encode(batch.candidate.bytes);
    let exact = HexPattern::new(&signature, &signature, 256).unwrap();
    let matched = engine.evaluate(&r, &exact, &message, 0, 3).unwrap();
    assert_eq!((matched.matches, matched.lane), (1, 0));
    assert_eq!(
        engine.evaluate(&r, &exact, &message, 1, 2).unwrap().matches,
        0
    );

    // Invalid request records must report errors, never release signatures.
    let mut bad = Vec::new();
    let mut c = r;
    c.salt_length = 223;
    bad.push(c);
    let mut c = r;
    c.source = 2;
    bad.push(c);
    let mut c = r;
    c.p[127] &= !1;
    bad.push(c);
    let mut c = r;
    c.q = c.p;
    bad.push(c);
    let mut c = r;
    c.dp[127] ^= 1;
    bad.push(c);
    let mut c = r;
    c.dq[127] ^= 1;
    bad.push(c);
    let mut c = r;
    c.q_inv[127] ^= 1;
    bad.push(c);
    let mut c = r;
    c.source = 1;
    c.offset = u64::MAX;
    c.length = 2;
    bad.push(c);
    let mut c = r;
    c.source = 1;
    c.length = 0;
    bad.push(c);
    for (index, c) in bad.into_iter().enumerate() {
        eprintln!("RSA-PSS malformed request={index}");
        assert!(engine.evaluate(&c, &any, &message, 0, 1).is_err());
    }
    let mut exhausted = r;
    exhausted.salt_length = 0;
    assert!(engine.evaluate(&exhausted, &any, &message, 1, 1).is_err());
    let before = engine.launches;
    for (start, count) in [(0, 0), (0, 4), (u64::MAX, 2)] {
        assert!(engine.evaluate(&r, &any, &message, start, count).is_err());
    }
    assert_eq!(engine.launches, before);

    // Winner reconstruction remains required when per-lane audit is disabled.
    let mut engine = RsaPssTransport::load(&artifacts(), 1, 1, false).unwrap();
    assert_eq!(
        engine.evaluate(&r, &exact, &message, 0, 1).unwrap().matches,
        1
    );
    assert_eq!(
        engine.evaluate(&r, &exact, &message, 1, 1).unwrap().matches,
        0
    );
}

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires freshly built RSA-PSS artifacts and Apple GPU"]
fn cli_both_sources_verify_exported_signatures_and_reject_seed() {
    let key = key();
    let dir = Directory(std::env::temp_dir().join(format!(
        "metal-pss-test-{}-{}",
        std::process::id(),
        rand::random::<u64>()
    )));
    std::fs::create_dir(&dir.0).unwrap();
    let private = dir.0.join("private.pem");
    let message = dir.0.join("message.bin");
    std::fs::write(
        &private,
        key.to_pkcs8_pem(LineEnding::LF).unwrap().as_bytes(),
    )
    .unwrap();
    std::fs::write(&message, b"header0000footer").unwrap();
    for source in ["salt", "message"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_vanity-miner"));
        command
            .args(["--metal-artifacts"])
            .arg(artifacts())
            .args([
                "--verify",
                "--batch-size",
                "1",
                "--threads-per-group",
                "1",
                "--batches",
                "2",
                "rsa-pss-signature-vanity",
                "--key",
            ])
            .arg(&private)
            .arg("--message")
            .arg(&message)
            .args(["--search-source", source, "--salt-length", "32"]);
        if source == "message" {
            command.args([
                "--nonce-offset",
                "6",
                "--nonce-length",
                "4",
                "--fixed-salt-hex",
                &"42".repeat(32),
            ]);
        }
        let out = command.output().unwrap();
        assert!(
            out.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(String::from_utf8_lossy(&out.stderr).contains("Metal RSA-PSS: 2 launches"));
        let mut signature = None;
        let mut salt = None;
        let mut verified = 0;
        for line in text.lines() {
            if let Some(value) = line.strip_prefix("[rsa-pss] signature=") {
                signature = Some(hex::decode(value).unwrap());
            }
            if let Some(value) = line.strip_prefix("[rsa-pss] salt=") {
                salt = Some(hex::decode(value).unwrap());
            }
            if let Some(value) = line.strip_prefix("[rsa-pss] message=") {
                let bytes = hex::decode(value).unwrap();
                let salt = salt.take().unwrap();
                key.to_public_key()
                    .verify(
                        Pss::new_with_salt::<Sha256>(salt.len()),
                        &Sha256::digest(&bytes),
                        &signature.take().unwrap(),
                    )
                    .unwrap();
                if source == "salt" {
                    assert_eq!(bytes, b"header0000footer");
                } else {
                    assert_eq!(&bytes[..6], b"header");
                    assert_eq!(&bytes[10..], b"footer");
                    assert_eq!(salt, vec![0x42; 32]);
                }
                verified += 1;
            }
        }
        assert_eq!(verified, 2);
        let out = command.args(["--seed", "1"]).output().unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("--seed is unsupported"));
    }
}
