//! Bounded P-256/SHA-256 signature searches with host winner verification.

use crate::search_control::SearchControl;
use logic::crypto::sha256::Sha256;
use logic::{
    crypto::p256_vanity::{
        candidate_scalar,
        signatures::{self, SForm, SignatureTarget},
    },
    search::crypto_search::{CandidateDeriver, CandidateDomain, write_message_counter},
    search::hex_pattern::HexPattern,
};
use p256::{
    SecretKey,
    ecdsa::{Signature, SigningKey, signature::Signer},
    elliptic_curve::sec1::ToEncodedPoint,
    pkcs8::DecodePrivateKey,
};
use rand::{RngCore, rngs::OsRng};
use std::{path::PathBuf, sync::Arc, thread, time::Duration};
use zeroize::Zeroizing;

#[derive(Clone, Copy)]
pub enum SearchSource {
    Message { offset: usize, length: usize },
    Ephemeral,
}

pub struct SignatureSearch {
    pub key: PathBuf,
    pub message: PathBuf,
    pub source: SearchSource,
    pub prefix: String,
    pub suffix: String,
    pub target: SignatureTarget,
    pub s_form: SForm,
    pub workers: usize,
}

pub struct SignatureReport {
    pub candidates_tested: u64,
    pub candidate_unit: &'static str,
    pub elapsed: Duration,
    pub found: bool,
    pub output: Option<String>,
}

struct Winner {
    counter: u64,
    worker: u64,
    signature: [u8; 64],
}

impl SignatureSearch {
    pub fn pattern(&self) -> Result<HexPattern, String> {
        let width = if self.target == SignatureTarget::Raw {
            64
        } else {
            32
        };
        HexPattern::new(&self.prefix, &self.suffix, width).map_err(|e| e.to_string())
    }

    pub fn validate(&self) -> Result<(), String> {
        self.pattern()?;
        if self.workers == 0 {
            return Err("worker count must be nonzero".into());
        }
        Ok(())
    }
}

pub fn run_cpu(
    config: &SignatureSearch,
    control: Arc<SearchControl>,
) -> Result<SignatureReport, String> {
    run(config, control, None)
}

pub fn run_device(
    config: &SignatureSearch,
    control: Arc<SearchControl>,
    device: &mut EvaluateBatch<'_>,
) -> Result<SignatureReport, String> {
    run(config, control, Some(device))
}

fn run(
    config: &SignatureSearch,
    control: Arc<SearchControl>,
    device: Option<&mut EvaluateBatch<'_>>,
) -> Result<SignatureReport, String> {
    config.validate()?;
    let pattern = config.pattern()?;
    let pem = Zeroizing::new(
        std::fs::read_to_string(&config.key).map_err(|_| "could not read private-key PEM")?,
    );
    let key = SecretKey::from_pkcs8_pem(&pem)
        .map_err(|_| "key must be an unencrypted PKCS#8 P-256 private key")?;
    drop(pem);
    let private = Zeroizing::new(<[u8; 32]>::from(key.to_bytes()));
    let signing_key = SigningKey::from(&key);
    let public: [u8; 65] = key
        .public_key()
        .to_encoded_point(false)
        .as_bytes()
        .try_into()
        .map_err(|_| "invalid public point encoding")?;
    let original = std::fs::read(&config.message).map_err(|_| "could not read message")?;
    if let SearchSource::Message { offset, length } = config.source {
        // Validate without altering the retained original message.
        let end = offset
            .checked_add(length)
            .ok_or("message window exceeds message bounds")?;
        if length == 0 || end > original.len() {
            return Err("message window must be nonempty and within the message".into());
        }
    }
    let digest: [u8; 32] = Sha256::digest(&original);
    let mut seed = Zeroizing::new([0; 32]);
    OsRng
        .try_fill_bytes(seed.as_mut())
        .map_err(|_| "OS cryptographic entropy unavailable")?;
    let deriver = CandidateDeriver::new(
        *seed,
        CandidateDomain::P256Ephemeral,
        Sha256::digest(public),
        digest,
    );

    let candidate_limit = match config.source {
        SearchSource::Message { length, .. } if length < 8 => 1u64 << (length * 8),
        _ => u64::MAX,
    };
    let outcome = if let Some(device) = device {
        use logic::modes::p256_signature_vanity::P256SignatureRequest;
        let (source, offset, length) = match config.source {
            SearchSource::Message { offset, length } => (0, offset as u64, length as u64),
            SearchSource::Ephemeral => (1, 0, 0),
        };
        let request = Zeroizing::new(P256SignatureRequest {
            private: *private,
            seed: *seed,
            fingerprint: Sha256::digest(public),
            digest,
            worker: 0,
            offset,
            length,
            source,
            target: match config.target {
                SignatureTarget::Raw => 0,
                SignatureTarget::R => 1,
                SignatureTarget::S => 2,
            },
            s_form: match config.s_form {
                SForm::Low => 0,
                SForm::High => 1,
                SForm::Either => 2,
            },
            reserved: 0,
        });
        let found = crate::search_batches::find(
            |start, count| device(&request, &pattern, &original, start, count),
            candidate_limit,
            &control,
            |counter, bytes| {
                let signature: [u8; 64] = bytes[..64]
                    .try_into()
                    .map_err(|_| "invalid device signature width")?;
                let mut message = original.clone();
                let reproduced = match config.source {
                    SearchSource::Message { offset, length } => {
                        write_message_counter(&mut message, offset, length, counter as u128)
                            .map_err(|e| e.to_string())?;
                        let raw = signatures::sign_message(&private, &message)
                            .ok_or("device message reconstruction failed")?;
                        signatures::matching_representation(
                            &raw,
                            config.target,
                            config.s_form,
                            &pattern,
                        )
                    }
                    SearchSource::Ephemeral => {
                        let nonce = candidate_scalar(&deriver, 0, counter as u128)
                            .ok_or("device nonce reconstruction failed")?;
                        signatures::matching_ephemeral_signature(
                            &private,
                            &digest,
                            &nonce,
                            config.target,
                            config.s_form,
                            &pattern,
                        )
                    }
                };
                if reproduced != Some(signature)
                    || !signatures::verify(&public, &message, &signature)
                {
                    return Err(
                        "device P-256 signature failed reconstruction and verification".into(),
                    );
                }
                Ok(true)
            },
        )?;
        found.map(|(counter, result)| Winner {
            counter,
            worker: 0,
            signature: result.bytes[..64]
                .try_into()
                .expect("fixed signature width"),
        })
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for worker in 0..config.workers {
                let (original, control, deriver, private, pattern, signing_key) = (
                    &original,
                    &control,
                    &deriver,
                    &private,
                    &pattern,
                    &signing_key,
                );
                handles.push(scope.spawn(move || -> Result<Option<Winner>, String> {
                    let stop_peers = control.cancel_on_exit();
                    let mut message = original.clone();
                    while let Some(batch) = control.reserve_bounded_batch(64, candidate_limit) {
                        for counter in batch {
                            if control.stopped() {
                                return Ok(None);
                            }
                            let matched = match config.source {
                                SearchSource::Message { offset, length } => {
                                    write_message_counter(
                                        &mut message,
                                        offset,
                                        length,
                                        counter as u128,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    let signature: Signature = signing_key
                                        .try_sign(&message)
                                        .map_err(|_| "deterministic P-256 signing failed")?;
                                    signatures::matching_representation(
                                        &signature.to_bytes().into(),
                                        config.target,
                                        config.s_form,
                                        pattern,
                                    )
                                }
                                SearchSource::Ephemeral => {
                                    let nonce =
                                        candidate_scalar(deriver, worker as u64, counter as u128)
                                            .ok_or("ephemeral scalar derivation exhausted")?;
                                    signatures::matching_ephemeral_signature(
                                        private,
                                        &digest,
                                        &nonce,
                                        config.target,
                                        config.s_form,
                                        pattern,
                                    )
                                }
                            };
                            control.add_tested(1);
                            let Some(signature) = matched else {
                                continue;
                            };
                            if !signatures::verify(&public, &message, &signature)
                                || !pattern.matches(config.target.bytes(&signature))
                            {
                                return Err("P-256 winning signature failed verification".into());
                            }
                            if control.claim_verified_winner() {
                                return Ok(Some(Winner {
                                    counter,
                                    worker: worker as u64,
                                    signature,
                                }));
                            }
                            return Ok(None);
                        }
                    }
                    stop_peers.finish();
                    Ok(None)
                }));
            }
            let mut winner = None;
            let mut error = None;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(Some(found))) => winner = Some(found),
                    Ok(Ok(None)) => {}
                    Ok(Err(message)) => {
                        control.cancel();
                        error = Some(message);
                    }
                    Err(_) => {
                        control.cancel();
                        error = Some("P-256 signature worker panicked".into());
                    }
                }
            }
            if let Some(error) = error {
                Err(error)
            } else {
                Ok(winner)
            }
        })?
    };
    if outcome.is_none() && !control.stopped() {
        return Err("signature search exhausted its unique candidate space without a match".into());
    }
    let found = outcome.is_some();
    let mut output = None;
    if let Some(winner) = outcome {
        // Reconstruct from metadata instead of trusting worker message buffers.
        let mut message = original.clone();
        let reproduced = match config.source {
            SearchSource::Message { offset, length } => {
                write_message_counter(&mut message, offset, length, winner.counter as u128)
                    .map_err(|e| e.to_string())?;
                let raw = signatures::sign_message(&private, &message)
                    .ok_or("winner signing reconstruction failed")?;
                signatures::matching_representation(&raw, config.target, config.s_form, &pattern)
            }
            SearchSource::Ephemeral => {
                let nonce = candidate_scalar(&deriver, winner.worker, winner.counter as u128)
                    .ok_or("winner nonce reconstruction failed")?;
                signatures::matching_ephemeral_signature(
                    &private,
                    &digest,
                    &nonce,
                    config.target,
                    config.s_form,
                    &pattern,
                )
            }
        };
        if reproduced != Some(winner.signature)
            || !signatures::verify(&public, &message, &winner.signature)
        {
            return Err("P-256 winner failed final reconstruction and verification".into());
        }
        output = Some(format!(
            "[p256-signature] public_key={}\n[p256-signature] signature={}\n[p256-signature] message={}",
            hex::encode(&public[1..]),
            hex::encode(winner.signature),
            hex::encode(&message),
        ));
    }
    if let Some(record) = &output {
        println!("{record}");
    }
    let (candidates_tested, elapsed) = control.statistics();
    Ok(SignatureReport {
        candidates_tested,
        candidate_unit: match config.source {
            SearchSource::Message { .. } => "messages",
            SearchSource::Ephemeral => "nonces",
        },
        elapsed,
        found,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
                run_device(&config, control, &mut crate::test_support::p256_signature)
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
            run_device(
                &exhausted,
                control.clone(),
                &mut crate::test_support::p256_signature,
            )
        } else {
            run_cpu(&exhausted, control.clone())
        };
        assert!(matches!(result, Err(error) if error.contains("exhausted")));
        assert_eq!(control.statistics().0, 256);
        assert!(!signature_out.exists());
        assert!(!message_out.unwrap().exists());
    }
}

/// Synchronized, ordered candidate evaluation for this mode. Implementations
/// must clear secret device buffers before returning; this is not a CPU fallback.
pub type EvaluateBatch<'a> = dyn FnMut(
        &logic::modes::p256_signature_vanity::P256SignatureRequest,
        &logic::search::hex_pattern::HexPattern,
        &[u8],
        u64,
        u32,
    ) -> Result<logic::search::candidate_result::BatchResult, String>
    + 'a;
