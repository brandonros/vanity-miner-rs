//! Explicit-salt RSA-PSS searches using blinded CPU private operations.

pub(crate) mod args;
pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;
#[cfg(feature = "cumetal")]
pub(crate) mod cumetal;

mod device;
#[cfg(test)]
mod tests;
mod verification;

use crate::{
    modes::rsa_keys::{fixed_bytes, validate_rsa2048},
    runner::session::SearchControl,
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

struct Prepared<'a> {
    config: &'a PssSearch,
    pattern: HexPattern,
    key: RsaPrivateKey,
    public: rsa::RsaPublicKey,
    original: Vec<u8>,
    digest: [u8; 32],
    base_salt: Vec<u8>,
    limit: u64,
}

impl<'a> Prepared<'a> {
    fn new(config: &'a PssSearch) -> Result<Self, String> {
        let pattern = config.validate()?;
        let pem = Zeroizing::new(
            std::fs::read_to_string(&config.key)
                .map_err(|_| "could not read RSA private-key PEM")?,
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
        Ok(Self {
            config,
            pattern,
            key,
            public,
            original,
            digest,
            base_salt,
            limit,
        })
    }
}

fn run(
    config: &PssSearch,
    control: Arc<SearchControl>,
    device: Option<&mut EvaluateBatch<'_>>,
) -> Result<PssReport, String> {
    let prepared = Prepared::new(config)?;
    let output = match device {
        Some(device) => prepared.search_device(&control, device)?,
        None => prepared
            .search_cpu(&control)?
            .map(|winner| prepared.format_winner(winner))
            .transpose()?,
    };
    if output.is_none() && !control.stopped() && !control.continuous() {
        return Err("RSA-PSS search exhausted its unique candidate space without a match".into());
    }
    let found = output.is_some();
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
        &logic::modes::rsa_pss::RsaPssRequest,
        &logic::search::hex_pattern::HexPattern,
        &[u8],
        u64,
        u32,
    ) -> Result<logic::search::candidate_result::BatchResult, String>
    + 'a;
