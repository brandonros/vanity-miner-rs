use super::*;
use logic::search::candidate_result::BatchResult;
use p256::pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding};
use std::path::PathBuf;

struct Cleanup(PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn bounded_search_exports_matching_interoperable_keys() {
    check_runner(false);
    check_runner(true);
}

fn check_runner(device: bool) {
    let mut random = [0; 16];
    OsRng.fill_bytes(&mut random);
    let dir = std::env::temp_dir().join(format!("vanity-p256-test-{}", hex::encode(random)));
    std::fs::create_dir(&dir).unwrap();
    let _cleanup = Cleanup(dir.clone());
    let private_out = dir.join("private.pem");
    let public_out = dir.join("public.pem");
    let mut config = PublicKeySearch {
        prefix: "a".into(),
        suffix: "".into(),
        target: PublicTarget::X,

        workers: 4,
    };
    if device {
        let mut corrupt = |_: &logic::modes::p256_public_key::P256PublicRequest,
                           _: &HexPattern,
                           _: &[u8],
                           _: u64,
                           _: u32| {
            let mut results = logic::search::candidate_result::BatchResult::EMPTY;
            results.matches = 1;
            results.lane = 0;
            results.candidate.status = 1; // Invalid all-zero SEC1 point, claimed as a winner.
            Ok(results)
        };
        let rejected = run_device(&config, Arc::new(SearchControl::new()), &mut corrupt);
        assert!(matches!(rejected, Err(error) if error.contains("failed verification")));
        assert!(!private_out.exists());
        assert!(!public_out.exists());
    }
    if device {
        let prefix = std::mem::take(&mut config.prefix);
        crate::modes::tests::assert_continuous(2, |control| {
            run_device(&config, control, &mut evaluate_batch).map(|_| ())
        });
        config.prefix = prefix;
    }
    let control = Arc::new(SearchControl::new());
    let report = if device {
        run_device(&config, control, &mut evaluate_batch)
    } else {
        run_cpu(&config, control)
    }
    .unwrap();
    assert!(report.found);
    assert!(report.candidates_tested > 0);
    let record = report.output.as_ref().unwrap();
    let private =
        SecretKey::from_slice(&crate::modes::tests::console_field(record, "private_key")).unwrap();
    let public = p256::PublicKey::from_sec1_bytes(&crate::modes::tests::console_field(
        record,
        "sec1_public_key",
    ))
    .unwrap();
    assert_eq!(
        crate::modes::tests::console_field(record, "public_key"),
        config.target.bytes(
            public
                .to_encoded_point(false)
                .as_bytes()
                .try_into()
                .unwrap()
        )
    );
    let private_pem = private.to_pkcs8_pem(LineEnding::LF).unwrap();
    let public_pem = public.to_public_key_pem(LineEnding::LF).unwrap();
    std::fs::write(&private_out, private_pem.as_bytes()).unwrap();
    std::fs::write(&public_out, &public_pem).unwrap();
    assert!(private.public_key() == public);
    use p256::elliptic_curve::sec1::ToEncodedPoint;
    assert_eq!(public.to_encoded_point(false).as_bytes()[1] >> 4, 0xa);
    assert!(config.validate().is_ok());
    // Check with an independent implementation when OpenSSL is installed.
    if std::process::Command::new("openssl")
        .arg("version")
        .output()
        .is_ok()
    {
        let result = std::process::Command::new("openssl")
            .args(["pkey", "-in"])
            .arg(&private_out)
            .arg("-pubout")
            .output()
            .unwrap();
        assert!(result.status.success());
        assert_eq!(result.stdout, public_pem.as_bytes());
        let message = dir.join("message.bin");
        let signature = dir.join("signature.der");
        std::fs::write(&message, b"public interoperability test message").unwrap();
        let signed = std::process::Command::new("openssl")
            .args(["dgst", "-sha256", "-sign"])
            .arg(&private_out)
            .arg(&message)
            .output()
            .unwrap();
        assert!(signed.status.success());
        std::fs::write(&signature, signed.stdout).unwrap();
        let verified = std::process::Command::new("openssl")
            .args(["dgst", "-sha256", "-verify"])
            .arg(&public_out)
            .arg("-signature")
            .arg(&signature)
            .arg(&message)
            .output()
            .unwrap();
        assert!(verified.status.success());
    }
    std::fs::remove_dir_all(dir).unwrap();
}
fn evaluate_batch(
    request: &logic::modes::p256_public_key::P256PublicRequest,
    pattern: &HexPattern,
    _message: &[u8],
    start: u64,
    count: u32,
) -> Result<BatchResult, String> {
    crate::modes::tests::evaluate_batch(start, count, |counter| {
        logic::modes::p256_public_key::p256_public(request, counter, pattern)
    })
}
