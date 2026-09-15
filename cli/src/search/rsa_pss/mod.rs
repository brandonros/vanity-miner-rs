//! Explicit-salt RSA-PSS searches using blinded CPU private operations.

use crate::{
    rsa_host::{fixed_bytes, validate_rsa2048},
    search_control::SearchControl,
};
use logic::crypto::sha256::Sha256;
use logic::{
    crypto::rsa_pss::encode_sha256,
    search::hex_pattern::HexPattern,
    search::{message_window::write_message_counter, salt_counter::write_salt_counter},
};
use rand::{RngCore, rngs::OsRng};
use rsa::{BigUint, Pss, RsaPrivateKey, pkcs8::DecodePrivateKey, traits::PublicKeyParts};
use std::{path::PathBuf, sync::Arc, thread, time::Duration};
use zeroize::Zeroizing;

#[derive(Clone)]
pub enum PssSource {
    Salt {
        length: usize,
    },
    Message {
        offset: usize,
        length: usize,
        fixed_salt: Option<Vec<u8>>,
        salt_length: usize,
    },
}

pub struct PssSearch {
    pub key: PathBuf,
    pub message: PathBuf,
    pub source: PssSource,
    pub prefix: String,
    pub suffix: String,
    pub workers: usize,
}

pub struct PssReport {
    pub candidates_tested: u64,
    pub candidate_unit: &'static str,
    pub elapsed: Duration,
    pub found: bool,
    pub output: Option<String>,
}
struct Winner {
    counter: u64,
    signature: [u8; 256],
}

pub fn sign_explicit_salt(
    key: &RsaPrivateKey,
    digest: &[u8; 32],
    salt: &[u8],
) -> Result<[u8; 256], String> {
    let mut encoded = [0; 256];
    encode_sha256(digest, salt, 2047, &mut encoded).map_err(|e| e.to_string())?;
    let representative = BigUint::from_bytes_be(&encoded);
    let signed = rsa::hazmat::rsa_decrypt_and_check(key, Some(&mut OsRng), &representative)
        .map_err(|_| "blinded RSA private operation failed")?;
    Ok(*fixed_bytes::<256>(&signed)?)
}

impl PssSearch {
    pub fn validate(&self) -> Result<HexPattern, String> {
        let pattern =
            HexPattern::new(&self.prefix, &self.suffix, 256).map_err(|e| e.to_string())?;
        if self.workers == 0 {
            return Err("worker count must be nonzero".into());
        }
        match &self.source {
            PssSource::Salt { length } if *length > 222 => {
                return Err("RSA-2048/SHA-256 permits at most 222 salt bytes".into());
            }
            PssSource::Message {
                fixed_salt,
                salt_length,
                ..
            } => {
                if *salt_length > 222 {
                    return Err("RSA-2048/SHA-256 permits at most 222 salt bytes".into());
                }
                if fixed_salt.as_ref().is_some_and(|salt| salt.len() > 222) {
                    return Err("RSA-2048/SHA-256 permits at most 222 salt bytes".into());
                }
            }
            _ => {}
        }
        Ok(pattern)
    }
}

fn apply_candidate(
    source: &PssSource,
    base_salt: &[u8],
    counter: u64,
    message: &mut [u8],
    salt: &mut [u8],
) -> Result<(), String> {
    match source {
        PssSource::Salt { .. } => write_salt_counter(base_salt, counter, salt),
        PssSource::Message { offset, length, .. } => {
            write_message_counter(message, *offset, *length, counter as u128)
        }
    }
    .map_err(|e| e.to_string())
}

pub fn run_cpu(config: &PssSearch, control: Arc<SearchControl>) -> Result<PssReport, String> {
    run(config, control, None)
}

pub fn run_device(
    config: &PssSearch,
    control: Arc<SearchControl>,
    device: &mut EvaluateBatch<'_>,
) -> Result<PssReport, String> {
    run(config, control, Some(device))
}

fn run(
    config: &PssSearch,
    control: Arc<SearchControl>,
    device: Option<&mut EvaluateBatch<'_>>,
) -> Result<PssReport, String> {
    let pattern = config.validate()?;
    let pem = Zeroizing::new(
        std::fs::read_to_string(&config.key).map_err(|_| "could not read RSA private-key PEM")?,
    );
    let mut key = RsaPrivateKey::from_pkcs8_pem(&pem)
        .map_err(|_| "key must be an unencrypted PKCS#8 RSA private key")?;
    drop(pem);
    validate_rsa2048(&mut key)?;
    if BigUint::from_bytes_be(pattern.minimum_value()) >= *key.n() {
        return Err("requested signature pattern lies outside the RSA modulus range".into());
    }
    let public = key.to_public_key();
    let original = std::fs::read(&config.message).map_err(|_| "could not read message")?;
    if let PssSource::Message { offset, length, .. } = &config.source
        && (*length == 0
            || offset
                .checked_add(*length)
                .is_none_or(|end| end > original.len()))
    {
        return Err("message window must be nonempty and within the message".into());
    }
    let digest: [u8; 32] = Sha256::digest(&original);
    let mut base_salt = match &config.source {
        PssSource::Salt { length } => vec![0; *length],
        PssSource::Message {
            fixed_salt: Some(salt),
            ..
        } => salt.clone(),
        PssSource::Message {
            fixed_salt: None,
            salt_length,
            ..
        } => vec![0; *salt_length],
    };
    if !matches!(
        config.source,
        PssSource::Message {
            fixed_salt: Some(_),
            ..
        }
    ) {
        OsRng
            .try_fill_bytes(&mut base_salt)
            .map_err(|_| "OS cryptographic entropy unavailable")?;
    }
    let variable_bytes = match config.source {
        PssSource::Salt { length } => length,
        PssSource::Message { length, .. } => length,
    };
    let limit = if variable_bytes < 8 {
        1u64 << (variable_bytes * 8)
    } else {
        u64::MAX
    };
    let outcome = if let Some(device) = device {
        use logic::modes::rsa_pss_signature_vanity::RsaPssRequest;
        use rsa::traits::PrivateKeyParts;
        let (source, offset, length) = match config.source {
            PssSource::Salt { .. } => (0, 0, 0),
            PssSource::Message { offset, length, .. } => (1, offset as u64, length as u64),
        };
        let coefficient =
            Zeroizing::new(key.crt_coefficient().ok_or("missing RSA CRT coefficient")?);
        let mut request = Zeroizing::new(RsaPssRequest {
            p: *fixed_bytes(&key.primes()[0])?,
            q: *fixed_bytes(&key.primes()[1])?,
            dp: *fixed_bytes(key.dp().ok_or("missing RSA dp")?)?,
            dq: *fixed_bytes(key.dq().ok_or("missing RSA dq")?)?,
            q_inv: *fixed_bytes(&coefficient)?,
            digest,
            salt: [0; 222],
            reserved: [0; 2],
            offset,
            length,
            source,
            salt_length: base_salt.len() as u32,
        });
        request.salt[..base_salt.len()].copy_from_slice(&base_salt);
        let found = crate::search_batches::find(
            |start, count| device(&request, &pattern, &original, start, count),
            limit,
            &control,
            |counter, signature| {
                let mut message = original.clone();
                let mut salt = base_salt.clone();
                apply_candidate(&config.source, &base_salt, counter, &mut message, &mut salt)?;
                let digest = Sha256::digest(&message);
                let reproduced = sign_explicit_salt(&key, &digest, &salt)?;
                if reproduced != *signature || !pattern.matches(signature) {
                    return Err("device RSA-PSS signature failed reconstruction".into());
                }
                public
                    .verify(Pss::new_with_salt::<Sha256>(salt.len()), &digest, signature)
                    .map_err(|_| "device RSA-PSS signature failed independent verification")?;
                Ok(true)
            },
        )?;
        found.map(|(counter, result)| Winner {
            counter,
            signature: result.bytes,
        })
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..config.workers {
                let (key, original, public, base_salt, control, pattern) =
                    (&key, &original, &public, &base_salt, &control, &pattern);
                handles.push(scope.spawn(move || -> Result<Option<Winner>, String> {
                    let stop_peers = control.cancel_on_exit();
                    let mut message = original.clone();
                    let mut salt = base_salt.clone();
                    while let Some(batch) = control.reserve_bounded_batch(16, limit) {
                        for counter in batch {
                            if control.stopped() {
                                return Ok(None);
                            }
                            apply_candidate(
                                &config.source,
                                base_salt,
                                counter,
                                &mut message,
                                &mut salt,
                            )?;
                            let candidate_digest =
                                if matches!(config.source, PssSource::Salt { .. }) {
                                    digest
                                } else {
                                    Sha256::digest(&message)
                                };
                            let signature = sign_explicit_salt(key, &candidate_digest, &salt)?;
                            control.add_tested(1);
                            if !pattern.matches(&signature) {
                                continue;
                            }
                            public
                                .verify(
                                    Pss::new_with_salt::<Sha256>(salt.len()),
                                    &candidate_digest,
                                    &signature,
                                )
                                .map_err(|_| "RSA-PSS winner failed independent verification")?;
                            if control.claim_verified_winner() {
                                return Ok(Some(Winner { counter, signature }));
                            }
                            return Ok(None);
                        }
                    }
                    stop_peers.finish();
                    Ok(None)
                }));
            }
            super::worker_results::join(handles, &control, "RSA-PSS worker panicked")
        })?
    };
    if outcome.is_none() && !control.stopped() {
        return Err("RSA-PSS search exhausted its unique candidate space without a match".into());
    }
    let found = outcome.is_some();
    let mut output = None;
    if let Some(winner) = outcome {
        let mut message = original.clone();
        let mut salt = base_salt.clone();
        apply_candidate(
            &config.source,
            &base_salt,
            winner.counter,
            &mut message,
            &mut salt,
        )?;
        let digest: [u8; 32] = Sha256::digest(&message);
        let reproduced = sign_explicit_salt(&key, &digest, &salt)?;
        if reproduced != winner.signature || !pattern.matches(&reproduced) {
            return Err("RSA-PSS winner failed reconstruction".into());
        }
        public
            .verify(
                Pss::new_with_salt::<Sha256>(salt.len()),
                &digest,
                &reproduced,
            )
            .map_err(|_| "RSA-PSS final verification failed")?;
        use rsa::traits::PublicKeyParts;
        output = Some(format!(
            "[rsa-pss] public_key={}\n[rsa-pss] signature={}\n[rsa-pss] salt={}\n[rsa-pss] message={}",
            hex::encode(public.n().to_bytes_be()),
            hex::encode(reproduced),
            hex::encode(&salt),
            hex::encode(&message),
        ));
    }
    if let Some(record) = &output {
        println!("{record}");
    }
    let (candidates_tested, elapsed) = control.statistics();
    Ok(PssReport {
        candidates_tested,
        candidate_unit: if matches!(config.source, PssSource::Salt { .. }) {
            "salts"
        } else {
            "messages"
        },
        elapsed,
        found,
        output,
    })
}

pub type EvaluateBatch<'a> = dyn FnMut(
        &logic::modes::rsa_pss_signature_vanity::RsaPssRequest,
        &logic::search::hex_pattern::HexPattern,
        &[u8],
        u64,
        u32,
    ) -> Result<logic::search::candidate_result::BatchResult, String>
    + 'a;

#[cfg(test)]
mod tests;
