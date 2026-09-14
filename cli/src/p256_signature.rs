//! Bounded P-256/SHA-256 signature searches with host winner verification.

use crate::{
    protected_output::{StagedOutput, validate_outputs},
    search_control::SearchControl,
};
use logic::{
    crypto_search::{CandidateDeriver, CandidateDomain, write_message_counter},
    hex_pattern::HexPattern,
    p256_vanity::{
        candidate_scalar,
        signatures::{self, SForm, SignatureTarget},
    },
};
use p256::{
    SecretKey,
    ecdsa::{Signature, SigningKey, signature::Signer},
    elliptic_curve::sec1::ToEncodedPoint,
    pkcs8::DecodePrivateKey,
};
use rand::{RngCore, rngs::OsRng};
use sha2::{Digest, Sha256};
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
    pub signature_out: PathBuf,
    pub message_out: Option<PathBuf>,
    pub der_out: Option<PathBuf>,
    pub force: bool,
    pub workers: usize,
}

pub struct SignatureReport {
    pub candidates_tested: u64,
    pub candidate_unit: &'static str,
    pub elapsed: Duration,
    pub found: bool,
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
        if matches!(self.source, SearchSource::Message { .. }) && self.message_out.is_none() {
            return Err("message search requires an explicit winning-message output path".into());
        }
        let mut outputs = vec![self.signature_out.as_path()];
        if let Some(path) = &self.message_out {
            outputs.push(path.as_path());
        }
        if let Some(path) = &self.der_out {
            outputs.push(path.as_path());
        }
        validate_outputs(&outputs, &[&self.key, &self.message], self.force)
            .map_err(|e| e.to_string())
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
    device: &mut dyn crate::device_search::DeviceSearch,
) -> Result<SignatureReport, String> {
    run(config, control, Some(device))
}

fn run(
    config: &SignatureSearch,
    control: Arc<SearchControl>,
    device: Option<&mut dyn crate::device_search::DeviceSearch>,
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
    let digest: [u8; 32] = Sha256::digest(&original).into();
    let mut seed = Zeroizing::new([0; 32]);
    OsRng
        .try_fill_bytes(seed.as_mut())
        .map_err(|_| "OS cryptographic entropy unavailable")?;
    let deriver = CandidateDeriver::new(
        *seed,
        CandidateDomain::P256Ephemeral,
        Sha256::digest(public).into(),
        digest,
    );

    let candidate_limit = match config.source {
        SearchSource::Message { length, .. } if length < 8 => 1u64 << (length * 8),
        _ => u64::MAX,
    };
    let outcome = if let Some(device) = device {
        use logic::device_search::P256SignatureRequest;
        let (source, offset, length) = match config.source {
            SearchSource::Message { offset, length } => (0, offset as u64, length as u64),
            SearchSource::Ephemeral => (1, 0, 0),
        };
        let request = Zeroizing::new(P256SignatureRequest {
            private: *private,
            seed: *seed,
            fingerprint: Sha256::digest(public).into(),
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
        let found = crate::device_search::find(
            device,
            &crate::device_search::Request::P256Signature(&request),
            &pattern,
            &original,
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
        let raw_stage = StagedOutput::new(&config.signature_out, &winner.signature, config.force)
            .map_err(|e| e.to_string())?;
        let message_stage = config
            .message_out
            .as_ref()
            .map(|path| StagedOutput::new(path, &message, config.force))
            .transpose()
            .map_err(|e| e.to_string())?;
        let der = Signature::from_slice(&winner.signature)
            .map_err(|_| "invalid signature encoding")?
            .to_der();
        let der_stage = config
            .der_out
            .as_ref()
            .map(|path| StagedOutput::new(path, der.as_bytes(), config.force))
            .transpose()
            .map_err(|e| e.to_string())?;
        // Stage every artifact before publishing; the raw signature is last.
        if let Some(staged) = message_stage {
            staged.publish().map_err(|e| e.to_string())?;
        }
        if let Some(staged) = der_stage {
            staged.publish().map_err(|e| e.to_string())?;
        }
        raw_stage.publish().map_err(|e| {
            format!("signature publication failed; companion outputs may exist: {e}")
        })?;
    }
    let (candidates_tested, elapsed) = control.statistics();
    Ok(SignatureReport {
        candidates_tested,
        candidate_unit: match config.source {
            SearchSource::Message { .. } => "messages tested",
            SearchSource::Ephemeral => "nonces tested",
        },
        elapsed,
        found,
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
        StagedOutput::new(&key_path, pem.as_bytes(), false)
            .unwrap()
            .publish()
            .unwrap();
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
            let config = SignatureSearch {
                key: key_path.clone(),
                message: message_path.clone(),
                source,
                prefix: "a".into(),
                suffix: "".into(),
                target: SignatureTarget::R,
                s_form: form,
                signature_out: dir.0.join(format!("signature-{i}.bin")),
                message_out: Some(dir.0.join(format!("message-{i}.bin"))),
                der_out: Some(dir.0.join(format!("signature-{i}.der"))),
                force: false,
                workers: 4,
            };
            let control = Arc::new(SearchControl::new());
            let report = if device {
                run_device(&config, control, &mut crate::device_search::HostDevice)
            } else {
                run_cpu(&config, control)
            }
            .unwrap();
            assert!(report.found);
            let raw: [u8; 64] = std::fs::read(&config.signature_out)
                .unwrap()
                .try_into()
                .unwrap();
            let winning_message = std::fs::read(config.message_out.as_ref().unwrap()).unwrap();
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
                    .arg(config.der_out.as_ref().unwrap())
                    .arg(config.message_out.as_ref().unwrap())
                    .output()
                    .unwrap();
                assert!(verified.status.success());
            }
            assert!(run_cpu(&config, Arc::new(SearchControl::new())).is_err());
        }
        // r cannot be zero in a valid ECDSA signature. Exhausting a one-byte
        // window must test all 256 distinct messages, including reserved tails.
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
            signature_out: dir.0.join("exhausted.bin"),
            message_out: Some(dir.0.join("exhausted-message.bin")),
            der_out: None,
            force: false,
            workers: 8,
        };
        let control = Arc::new(SearchControl::new());
        let result = if device {
            run_device(
                &exhausted,
                control.clone(),
                &mut crate::device_search::HostDevice,
            )
        } else {
            run_cpu(&exhausted, control.clone())
        };
        assert!(matches!(result, Err(error) if error.contains("exhausted")));
        assert_eq!(control.statistics().0, 256);
        assert!(!exhausted.signature_out.exists());
        assert!(!exhausted.message_out.unwrap().exists());
    }
}
