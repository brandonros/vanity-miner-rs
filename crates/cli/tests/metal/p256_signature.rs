#![cfg(all(feature = "metal", feature = "p256-signature", target_os = "macos"))]
use logic::{
    crypto::{
        p256::{
            public_point,
            signatures::{self, SignatureTarget},
        },
        sha256::Sha256,
    },
    modes::p256_signature::{P256SignatureRequest, p256_signature},
    search::{hex_pattern::HexPattern, message_window::write_message_counter},
};
use std::path::PathBuf;
use vanity_miner::modes::p256_signature::metal::P256SignatureTransport;

fn artifacts() -> PathBuf {
    super::support::artifacts("VANITY_METAL_P256_SIGNATURE_ARTIFACTS", "p256-signature")
}
fn request(message: &[u8], source: u32, target: u32, s_form: u32) -> P256SignatureRequest {
    // Published RFC 6979 A.2.5 example, never a generated/user private key.
    let private = hex::decode("c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721")
        .unwrap()
        .try_into()
        .unwrap();
    P256SignatureRequest {
        private,
        seed: [0x42; 32],
        fingerprint: Sha256::digest(public_point(&private).unwrap()),
        digest: Sha256::digest(message),
        worker: 3,
        offset: 0,
        length: 1,
        source,
        target,
        s_form,
        reserved: 0,
    }
}

#[test]
fn invalid_dispatch_is_rejected_before_loading_artifacts() {
    assert!(
        P256SignatureTransport::load(&PathBuf::from("missing"), 0, 1, true)
            .err()
            .unwrap()
            .contains("dispatch")
    );
}

#[test]
#[ignore = "build scripts/metal.sh build p256-signature first; requires Apple GPU"]
fn sources_targets_s_forms_and_variable_messages_match_cpu() {
    let message = b"sample";
    let mut engine = P256SignatureTransport::load(&artifacts(), 3, 4, true).unwrap();
    // The same prepared pipeline must survive empty/growing/shrinking payloads,
    // stale output from a preceding winner, and a rejected request.
    let mut expected_allocations = 1;
    let mut payload_capacity = 1;
    for length in [0usize, 1, 65, 0, 64, 129, 2, 129] {
        let message = vec![b'x'; length];
        let r = request(&message, 1, 0, 0);
        let pattern = HexPattern::new("", "", 64).unwrap();
        if length > payload_capacity {
            payload_capacity = length.next_power_of_two();
            expected_allocations += 1;
        }
        assert_eq!(
            engine
                .evaluate(&r, &pattern, &message, 0, 1)
                .unwrap()
                .matches,
            1
        );
        assert_eq!(engine.allocation_rounds, expected_allocations);
        let mut invalid = r;
        invalid.source = u32::MAX;
        assert!(engine.evaluate(&invalid, &pattern, &message, 0, 1).is_err());
        assert_eq!(engine.allocation_rounds, expected_allocations);
    }

    for source in [0, 1] {
        for (target, representation) in
            [SignatureTarget::Raw, SignatureTarget::R, SignatureTarget::S]
                .into_iter()
                .enumerate()
        {
            let width = if target == 0 { 64 } else { 32 };
            for form in [0, 1, 2] {
                eprintln!("P-256 signature source={source} target={target} s_form={form}");
                let request = request(message, source, target as u32, form);
                let all = HexPattern::new("", "", width).unwrap();
                let result = engine.evaluate(&request, &all, message, 0x72, 3).unwrap();
                assert_eq!(result.matches, 3);
                let mut candidate_message = message.to_vec();
                if source == 0 {
                    write_message_counter(
                        &mut candidate_message,
                        0,
                        1,
                        0x72 + u128::from(result.lane),
                    )
                    .unwrap();
                }
                let raw = result.candidate.bytes[..64].try_into().unwrap();
                assert!(signatures::verify(
                    &public_point(&request.private).unwrap(),
                    &candidate_message,
                    &raw
                ));
                let normalized = p256::ecdsa::Signature::from_slice(&raw)
                    .unwrap()
                    .normalize_s();
                if form == 0 {
                    assert!(normalized.is_none());
                }
                if form == 1 {
                    assert!(normalized.is_some());
                }
                let expected = p256_signature(&request, message, 0x73, &all);
                let raw: [u8; 64] = expected.bytes[..64].try_into().unwrap();
                if source == 0 {
                    // This exact counter restores "sample", the RFC known-answer input.
                    assert_eq!(
                        hex::encode(&raw[..32]),
                        "efd48b2aacb6a8fd1140dd9cd45e81d69d2c877b56aaf991c34d0ea84eaf3716"
                    );
                    if form == 1 {
                        assert_eq!(
                            hex::encode(&raw[32..]),
                            "f7cb1c942d657c41d436c7a1b6e29f65f3e900dbb9aff4064dc4ab2f843acda8"
                        );
                    }
                }
                let text = hex::encode(representation.bytes(&raw));
                let exact = HexPattern::new(&text, &text, width).unwrap();
                let winner = engine.evaluate(&request, &exact, message, 0x72, 3).unwrap();
                assert_eq!((winner.matches, winner.lane), (1, 1));
                assert_eq!(winner.candidate.bytes, expected.bytes);
                let impossible = HexPattern::new(&"0".repeat(width * 2), "", width).unwrap();
                assert_eq!(
                    engine
                        .evaluate(&request, &impossible, message, 0x72, 3)
                        .unwrap()
                        .matches,
                    0
                );
            }
        }
    }
    for length in [55, 56, 63, 64, 65, 127, 128, 129] {
        eprintln!("P-256 signature message length={length}");
        let message = vec![0x61; length];
        let mut request = request(&message, 0, 0, 0);
        request.offset = (length - 8) as u64;
        request.length = 8;
        assert_eq!(
            engine
                .evaluate(
                    &request,
                    &HexPattern::new("", "", 64).unwrap(),
                    &message,
                    u64::MAX - 1,
                    2
                )
                .unwrap()
                .matches,
            2
        );
    }
    let all = HexPattern::new("", "", 64).unwrap();
    for change in 0..4 {
        let mut request = request(message, 0, 0, 0);
        match change {
            0 => request.source = 2,
            1 => request.target = 3,
            2 => request.s_form = 3,
            _ => request.length = 7,
        }
        assert_eq!(
            engine
                .evaluate(&request, &all, message, 0, 1)
                .err()
                .as_deref(),
            Some("device candidate evaluation failed")
        );
    }
    let before = engine.launches;
    for (start, count) in [(0, 0), (0, 4), (u64::MAX, 2)] {
        assert!(
            engine
                .evaluate(&request(message, 0, 0, 0), &all, message, start, count)
                .is_err()
        );
    }
    assert_eq!(before, engine.launches);
    let mut no_audit = P256SignatureTransport::load(&artifacts(), 1, 1, false).unwrap();
    assert_eq!(
        no_audit
            .evaluate(&request(message, 1, 0, 0), &all, message, 0, 1)
            .unwrap()
            .matches,
        1
    );
}

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
#[ignore = "build scripts/metal.sh build p256-signature first; requires Apple GPU"]
fn bounded_cli_verifies_both_sources_and_s_forms() {
    use p256::pkcs8::{EncodePrivateKey, LineEnding};
    let directory = Directory(std::env::temp_dir().join(format!(
        "vanity-metal-p256-{}-{}",
        std::process::id(),
        rand::random::<u64>()
    )));
    std::fs::create_dir(&directory.0).unwrap();
    let key_path = directory.0.join("public-test-key.pem");
    let message_path = directory.0.join("message.bin");
    let message = b"header\0\0footer";
    let request = request(message, 0, 0, 0);
    let key = p256::SecretKey::from_slice(&request.private).unwrap();
    std::fs::write(
        &key_path,
        key.to_pkcs8_pem(LineEnding::LF).unwrap().as_bytes(),
    )
    .unwrap();
    std::fs::write(&message_path, message).unwrap();
    for source in ["message", "ephemeral"] {
        for form in ["low", "high", "either"] {
            let mut command = super::support::command(&artifacts());
            command.args([
                "--batches",
                "2",
                "--batch-size",
                "1",
                "--threads-per-group",
                "1",
                "--verify",
                "p256-signature-vanity",
                "--key",
                key_path.to_str().unwrap(),
                "--message",
                message_path.to_str().unwrap(),
                "--search-source",
                source,
                "--s-form",
                form,
            ]);
            if source == "message" {
                command.args(["--nonce-offset", "6", "--nonce-length", "2"]);
            }
            let result = command.output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let stdout = String::from_utf8(result.stdout).unwrap();
            let fields = |field: &str| -> Vec<Vec<u8>> {
                stdout
                    .lines()
                    .filter_map(|line| line.strip_prefix(&format!("[p256-signature] {field}=")))
                    .map(|value| hex::decode(value).unwrap())
                    .collect()
            };
            let signatures = fields("signature");
            let messages = fields("message");
            assert_eq!(signatures.len(), 2);
            assert_eq!(messages.len(), 2);
            for (raw, found) in signatures.iter().zip(&messages) {
                assert!(signatures::verify(
                    &public_point(&request.private).unwrap(),
                    found,
                    &raw.as_slice().try_into().unwrap()
                ));
                assert_eq!(&found[..6], &message[..6]);
                assert_eq!(&found[8..], &message[8..]);
                if source == "ephemeral" {
                    assert_eq!(found, message);
                }
                let normalized = p256::ecdsa::Signature::from_slice(raw)
                    .unwrap()
                    .normalize_s();
                if form == "low" {
                    assert!(normalized.is_none());
                }
                if form == "high" {
                    assert!(normalized.is_some());
                }
            }
        }
    }
    let rejected = super::support::command(&artifacts())
        .args([
            "--seed",
            "1",
            "--batches",
            "1",
            "p256-signature-vanity",
            "--key",
            key_path.to_str().unwrap(),
            "--message",
            message_path.to_str().unwrap(),
            "--search-source",
            "ephemeral",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("--seed is not supported"));
}
