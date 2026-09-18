//! Bounded P-256/SHA-256 signature searches with host winner verification.

pub(crate) mod args;
pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;
#[cfg(feature = "cumetal")]
pub(crate) mod cumetal;

#[cfg(feature = "metal")]
pub(crate) mod metal;

mod device;
#[cfg(test)]
mod tests;
mod verification;

use crate::runner::session::SearchControl;
use logic::crypto::sha256::Sha256;
use logic::{
    crypto::p256::{
        candidate_scalar,
        signatures::{self, SForm, SignatureTarget},
    },
    search::hex_pattern::HexPattern,
    search::{
        candidate_derivation::{CandidateDeriver, CandidateDomain},
        message_window::write_message_counter,
    },
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

struct Prepared<'a> {
    config: &'a SignatureSearch,
    pattern: HexPattern,
    private: Zeroizing<[u8; 32]>,
    signing_key: SigningKey,
    public: [u8; 65],
    original: Vec<u8>,
    digest: [u8; 32],
    seed: Zeroizing<[u8; 32]>,
    deriver: CandidateDeriver,
    candidate_limit: u64,
}

impl<'a> Prepared<'a> {
    fn new(config: &'a SignatureSearch) -> Result<Self, String> {
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
        Ok(Self {
            config,
            pattern,
            private,
            signing_key,
            public,
            original,
            digest,
            seed,
            deriver,
            candidate_limit,
        })
    }
}

fn run(
    config: &SignatureSearch,
    control: Arc<SearchControl>,
    device: Option<&mut EvaluateBatch<'_>>,
) -> Result<SignatureReport, String> {
    let prepared = Prepared::new(config)?;
    let output = match device {
        Some(device) => prepared.search_device(&control, device)?,
        None => prepared
            .search_cpu(&control)?
            .map(|winner| prepared.format_winner(winner))
            .transpose()?,
    };
    if output.is_none() && !control.stopped() && !control.continuous() {
        return Err("signature search exhausted its unique candidate space without a match".into());
    }
    let found = output.is_some();
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

pub type EvaluateBatch<'a> = dyn FnMut(
        &logic::modes::p256_signature::P256SignatureRequest,
        &logic::search::hex_pattern::HexPattern,
        &[u8],
        u64,
        u32,
    ) -> Result<logic::search::candidate_result::BatchResult, String>
    + 'a;
