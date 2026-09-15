//! Constructive RSA-2048 modulus search: interval and residue constraints on q.

use crate::{
    rsa_host::{fixed_bytes, sufficiently_separated, validate_rsa2048},
    search_control::SearchControl,
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

mod constraints;
use constraints::ceil_div;
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
    run(config, control, None)
}

pub fn run_device(
    config: &ModulusSearch,
    control: Arc<SearchControl>,
    device: &mut EvaluateBatch<'_>,
) -> Result<ModulusReport, String> {
    run(config, control, Some(device))
}

fn run(
    config: &ModulusSearch,
    control: Arc<SearchControl>,
    device: Option<&mut EvaluateBatch<'_>>,
) -> Result<ModulusReport, String> {
    let constraints = config.validate()?;
    let outcome = if let Some(device) = device {
        device::construct_device(&constraints, &control, device)?
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..config.workers {
                handles.push(scope.spawn(|| cpu::construct_worker(&constraints, &control)));
            }
            super::worker_results::join(handles, &control, "RSA modulus worker panicked")
        })?
    };
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

pub type EvaluateBatch<'a> = dyn FnMut(
        &logic::modes::rsa_modulus_vanity::RsaModulusRequest,
        &logic::search::hex_pattern::HexPattern,
        &[u8],
        u64,
        u32,
    ) -> Result<logic::search::candidate_result::BatchResult, String>
    + 'a;

#[cfg(test)]
mod tests;

mod cpu;

mod device;
