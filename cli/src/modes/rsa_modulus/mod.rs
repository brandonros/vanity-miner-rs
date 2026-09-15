//! Constructive RSA-2048 modulus search: interval and residue constraints on q.

pub(crate) mod args;
pub(crate) mod cpu;
#[cfg(feature = "gpu")]
pub(crate) mod cuda;
#[cfg(feature = "cumetal")]
pub(crate) mod cumetal;

mod constraints;
pub mod pipeline;
#[cfg(test)]
mod tests;

use crate::{
    modes::rsa_keys::{fixed_bytes, sufficiently_separated, validate_rsa2048},
    runner::session::SearchControl,
};
use logic::crypto::sha256::Sha256;
use logic::search::hex_pattern::HexPattern;
use num_bigint_dig::{BigUint, ModInverse, RandBigInt, prime::probably_prime};
use rand::rngs::OsRng;
use rsa::{Pss, RsaPrivateKey, pkcs8::EncodePrivateKey, traits::PublicKeyParts};
use std::{sync::Arc, thread, time::Duration};
use zeroize::Zeroizing;

pub struct ModulusSearch {
    pub prefix: String,
    pub suffix: String,
    pub workers: usize,
}

pub struct ModulusReport {
    pub q_candidates_tested: u64,
    pub elapsed: Duration,
    pub found: bool,
    pub output: Option<String>,
}

pub use constraints::{ModulusConstraints, QProgression};

impl ModulusSearch {
    pub fn validate(&self) -> Result<ModulusConstraints, String> {
        let constraints = ModulusConstraints::new(&self.prefix, &self.suffix)?;
        if self.workers == 0 {
            return Err("worker count must be nonzero".into());
        }
        Ok(constraints)
    }
}

pub fn run_cpu(
    config: &ModulusSearch,
    control: Arc<SearchControl>,
) -> Result<ModulusReport, String> {
    let constraints = config.validate()?;
    let outcome = thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..config.workers {
            handles.push(scope.spawn(|| cpu::construct_worker(&constraints, &control)));
        }
        crate::runner::workers::join(handles, &control, "RSA modulus worker panicked")
    })?;
    let found = outcome.is_some();
    let mut output = None;
    if let Some(key) = outcome {
        let private = key
            .to_pkcs8_der()
            .map_err(|_| "RSA private key encoding failed")?;
        output = Some(format!(
            "[rsa-modulus] public_key={}\n[rsa-modulus] public_exponent={}\n[rsa-modulus] private_key_pkcs8={}",
            hex::encode(key.n().to_bytes_be()),
            hex::encode(key.e().to_bytes_be()),
            hex::encode(private.as_bytes()),
        ));
    }
    if let Some(record) = &output {
        println!("{record}");
    }
    let (q_candidates_tested, elapsed) = control.statistics();
    Ok(ModulusReport {
        q_candidates_tested,
        elapsed,
        found,
        output,
    })
}
