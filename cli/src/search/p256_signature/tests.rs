use super::*;
use logic::search::candidate_result::BatchResult;
use p256::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn both_sources_emit_verified_signatures_and_exact_messages() {
    check_runner(false);
    check_runner(true);
}

fn check_runner(device: bool) {
    let mut random = [0; 16];
    OsRng.fill_bytes(&mut random);
    let dir = Directory(
        std::env::temp_dir().join(format!("vanity-signature-test-{}", hex::encode(random))),
    );
    std::fs::create_dir(&dir.0).unwrap();
    let key = SecretKey::random(&mut OsRng);
    let pem = key.to_pkcs8_pem(LineEnding::LF).unwrap();
    let key_path = dir.0.join("key.pem");
    std::fs::write(&key_path, pem.as_bytes()).unwrap();
    let public_path = dir.0.join("public.pem");
    std::fs::write(
        &public_path,
        key.public_key().to_public_key_pem(LineEnding::LF).unwrap(),
    )
    .unwrap();
    let message_path = dir.0.join("message.bin");
    let message = b"header\0\0\0\0footer";
    std::fs::write(&message_path, message).unwrap();
    let public = key
        .public_key()
        .to_encoded_point(false)
        .as_bytes()
        .try_into()
        .unwrap();
    for (i, (source, form)) in [
        (
            SearchSource::Message {
                offset: 6,
                length: 4,
            },
            SForm::Low,
        ),
        (SearchSource::Ephemeral, SForm::High),
        (SearchSource::Ephemeral, SForm::Either),
    ]
    .into_iter()
    .enumerate()
    {
        let signature_out = dir.0.join(format!("signature-{i}.bin"));
        let message_out = Some(dir.0.join(format!("message-{i}.bin")));
        let der_out = Some(dir.0.join(format!("signature-{i}.der")));
        let config = SignatureSearch {
            key: key_path.clone(),
            message: message_path.clone(),
            source,
            prefix: "a".into(),
            suffix: "".into(),
            target: SignatureTarget::R,
            s_form: form,

            workers: 4,
        };
        let control = Arc::new(SearchControl::new());
        let report = if device {
            run_device(&config, control, &mut evaluate_batch)
        } else {
            run_cpu(&config, control)
        }
        .unwrap();
        assert!(report.found);
        let record = report.output.as_ref().unwrap();
        let raw: [u8; 64] = crate::test_support::console_field(record, "signature")
            .try_into()
            .unwrap();
        let winning_message = crate::test_support::console_field(record, "message");
        std::fs::write(&signature_out, raw).unwrap();
        std::fs::write(message_out.as_ref().unwrap(), &winning_message).unwrap();
        std::fs::write(
            der_out.as_ref().unwrap(),
            Signature::from_slice(&raw).unwrap().to_der().as_bytes(),
        )
        .unwrap();

        assert!(signatures::verify(&public, &winning_message, &raw));
        assert_eq!(raw[0] >> 4, 0xa);
        let parsed = Signature::from_slice(&raw).unwrap();
        match form {
            SForm::Low => assert!(parsed.normalize_s().is_none()),
            SForm::High => assert!(parsed.normalize_s().is_some()),
            SForm::Either => {}
        }
        assert_eq!(&winning_message[..6], &message[..6]);
        assert_eq!(&winning_message[10..], &message[10..]);
        if matches!(source, SearchSource::Ephemeral) {
            assert_eq!(winning_message, message);
        }
        if std::process::Command::new("openssl")
            .arg("version")
            .output()
            .is_ok()
        {
            let verified = std::process::Command::new("openssl")
                .args(["dgst", "-sha256", "-verify"])
                .arg(&public_path)
                .arg("-signature")
                .arg(der_out.as_ref().unwrap())
                .arg(message_out.as_ref().unwrap())
                .output()
                .unwrap();
            assert!(verified.status.success());
        }
        assert!(config.validate().is_ok());
    }
    // r cannot be zero in a valid ECDSA signature. Exhausting a one-byte
    // window must test all 256 distinct messages, including reserved tails.
    let signature_out = dir.0.join("exhausted.bin");
    let message_out = Some(dir.0.join("exhausted-message.bin"));
    let _der_out: Option<PathBuf> = None;
    let exhausted = SignatureSearch {
        key: key_path,
        message: message_path,
        source: SearchSource::Message {
            offset: 6,
            length: 1,
        },
        prefix: "0".repeat(64),
        suffix: "".into(),
        target: SignatureTarget::R,
        s_form: SForm::Low,

        workers: 8,
    };
    let control = Arc::new(SearchControl::new());
    let result = if device {
        run_device(&exhausted, control.clone(), &mut evaluate_batch)
    } else {
        run_cpu(&exhausted, control.clone())
    };
    assert!(matches!(result, Err(error) if error.contains("exhausted")));
    assert_eq!(control.statistics().0, 256);
    assert!(!signature_out.exists());
    assert!(!message_out.unwrap().exists());
}
fn evaluate_batch(
    request: &logic::modes::p256_signature_vanity::P256SignatureRequest,
    pattern: &HexPattern,
    message: &[u8],
    start: u64,
    count: u32,
) -> Result<BatchResult, String> {
    crate::test_support::evaluate(start, count, |counter| {
        logic::modes::p256_signature_vanity::p256_signature(request, message, counter, pattern)
    })
}
