use super::*;
use logic::search::candidate_result::BatchResult;
use rsa::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};

struct Directory(PathBuf);
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn both_sources_produce_reproducible_pss_signatures() {
    check_runner(false);
    check_runner(true);
}

fn check_runner(device: bool) {
    let mut random = [0; 16];
    OsRng.fill_bytes(&mut random);
    let dir =
        Directory(std::env::temp_dir().join(format!("vanity-pss-test-{}", hex::encode(random))));
    std::fs::create_dir(&dir.0).unwrap();
    let key = RsaPrivateKey::new(&mut OsRng, 2048).unwrap();
    let pem = key.to_pkcs8_pem(LineEnding::LF).unwrap();
    let key_path = dir.0.join("private.pem");
    std::fs::write(&key_path, pem.as_bytes()).unwrap();
    let public_path = dir.0.join("public.pem");
    std::fs::write(
        &public_path,
        key.to_public_key()
            .to_public_key_pem(LineEnding::LF)
            .unwrap(),
    )
    .unwrap();
    let message_path = dir.0.join("input.bin");
    let message = b"header\0\0\0\0footer";
    std::fs::write(&message_path, message).unwrap();
    let sources = [
        PssSource::Salt { length: 32 },
        PssSource::Message {
            offset: 6,
            length: 4,
            fixed_salt: Some(vec![0x42; 32]),
            salt_length: 32,
        },
        PssSource::Message {
            offset: 6,
            length: 4,
            fixed_salt: None,
            salt_length: 32,
        },
        PssSource::Salt { length: 0 },
    ];
    for (i, source) in sources.into_iter().enumerate() {
        let signature_out = dir.0.join(format!("signature-{i}.bin"));
        let salt_out = dir.0.join(format!("salt-{i}.bin"));
        let message_out = Some(dir.0.join(format!("message-{i}.bin")));
        let config = PssSearch {
            key: key_path.clone(),
            message: message_path.clone(),
            source,
            prefix: if i == 3 { "" } else { "0" }.into(),
            suffix: "".into(),

            workers: 2,
        };
        let control = Arc::new(SearchControl::new());
        let report = if device {
            run_device(&config, control, &mut evaluate_batch)
        } else {
            run_cpu(&config, control)
        }
        .unwrap();
        assert!(report.found);
        if i == 3 {
            assert_eq!(report.candidates_tested, 1);
        }
        let record = report.output.as_ref().unwrap();
        let signature: [u8; 256] = crate::test_support::console_field(record, "signature")
            .try_into()
            .unwrap();
        let salt = crate::test_support::console_field(record, "salt");
        let winning_message = crate::test_support::console_field(record, "message");
        std::fs::write(&signature_out, signature).unwrap();
        std::fs::write(&salt_out, &salt).unwrap();
        std::fs::write(message_out.as_ref().unwrap(), &winning_message).unwrap();
        if matches!(config.source, PssSource::Salt { .. }) {
            assert_eq!(winning_message, message);
        }
        assert_eq!(&winning_message[..6], &message[..6]);
        assert_eq!(&winning_message[10..], &message[10..]);
        if i == 1 {
            assert_eq!(salt, vec![0x42; 32]);
        }
        let digest: [u8; 32] = Sha256::digest(&winning_message);
        key.to_public_key()
            .verify(
                Pss::new_with_salt::<Sha256>(salt.len()),
                &digest,
                &signature,
            )
            .unwrap();
        assert_eq!(sign_explicit_salt(&key, &digest, &salt).unwrap(), signature);
        if std::process::Command::new("openssl")
            .arg("version")
            .output()
            .is_ok()
        {
            let result = std::process::Command::new("openssl")
                .args(["dgst", "-sha256", "-verify"])
                .arg(&public_path)
                .arg("-signature")
                .arg(&signature_out)
                .args(["-sigopt", "rsa_padding_mode:pss", "-sigopt"])
                .arg(format!("rsa_pss_saltlen:{}", salt.len()))
                .arg(message_out.as_ref().unwrap())
                .output()
                .unwrap();
            assert!(result.status.success());
        }
    }
}
fn evaluate_batch(
    request: &logic::modes::rsa_pss_signature_vanity::RsaPssRequest,
    pattern: &HexPattern,
    message: &[u8],
    start: u64,
    count: u32,
) -> Result<BatchResult, String> {
    crate::test_support::evaluate(start, count, |counter| {
        logic::modes::rsa_pss_signature_vanity::rsa_pss(request, message, counter, pattern)
    })
}
