//! Constructive RSA-2048 modulus search: interval and residue constraints on q.

pub(crate) mod args;
pub(crate) mod cpu;

#[cfg(feature = "metal")]
pub mod metal;

mod constraints;
pub mod device;
#[cfg(test)]
mod tests;

use crate::{
    modes::rsa_keys::{fixed_bytes, sufficiently_separated, validate_rsa2048},
    runner::session::SearchControl,
};
use logic::crypto::sha256::Sha256;
use logic::search::hex_pattern::HexPattern;
#[cfg(test)]
use num_bigint_dig::RandBigInt;
use num_bigint_dig::{BigUint, ModInverse};
use rand::rngs::OsRng;
use rsa::{Pss, RsaPrivateKey, pkcs8::EncodePrivateKey, traits::PublicKeyParts};
use std::{sync::Arc, time::Duration};
use zeroize::Zeroizing;

pub struct ModulusSearch {
    pub prefix: String,
    pub suffix: String,
    pub workers: usize,
}

pub struct ModulusReport {
    pub candidates_tested: u64,
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
    use rand::RngCore;
    let constraints = config.validate()?;
    let mut seed = Zeroizing::new([0; 32]);
    OsRng
        .try_fill_bytes(seed.as_mut())
        .map_err(|_| "OS cryptographic entropy unavailable")?;
    let mining_config = Zeroizing::new(constraints.device_config(*seed, 0)?);
    let output = crate::runner::workers::search(config.workers, &control, |_| {
        cpu::construct_worker(&mining_config, &constraints.pattern, &control)
    })?;
    let found = output.is_some();
    if let Some(record) = &output {
        crate::runner::progress::print_verified(&control, record.clone())?;
    }
    let (candidates_tested, elapsed) = control.statistics();
    Ok(ModulusReport {
        candidates_tested,
        elapsed,
        found,
        output,
    })
}
