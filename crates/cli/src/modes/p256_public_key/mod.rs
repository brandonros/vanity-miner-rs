//! Bounded host runner for P-256 public-key searches.

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

use crate::runner::session::SearchControl;
use logic::{
    crypto::p256::{PublicTarget, candidate_scalar, public_point},
    search::candidate_derivation::{CandidateDeriver, CandidateDomain},
    search::hex_pattern::HexPattern,
};
use p256::SecretKey;
use rand::{RngCore, rngs::OsRng};
use std::{sync::Arc, thread, time::Duration};
use zeroize::Zeroizing;

pub struct PublicKeySearch {
    pub prefix: String,
    pub suffix: String,
    pub target: PublicTarget,
    pub workers: usize,
}

pub struct SearchReport {
    pub candidates_tested: u64,
    pub elapsed: Duration,
    pub found: bool,
    pub output: Option<String>,
}

struct Winner {
    private: Zeroizing<[u8; 32]>,
    public: [u8; 65],
}

impl PublicKeySearch {
    pub fn pattern(&self) -> Result<HexPattern, String> {
        let mut pattern = HexPattern::new(&self.prefix, &self.suffix, self.target.width())
            .map_err(|e| e.to_string())?;
        if self.target == PublicTarget::Uncompressed {
            pattern
                .constrain_byte(0, 255, 4)
                .map_err(|e| e.to_string())?;
        }
        Ok(pattern)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.pattern()?;
        if self.workers == 0 {
            return Err("worker count must be nonzero".into());
        }
        Ok(())
    }
}

use verification::verify_winner;

/// All workers join before output. The passed control also permits cancellation
/// by a host Ctrl-C handler; this library does not install a process-wide handler.
pub fn run_cpu(
    config: &PublicKeySearch,
    control: Arc<SearchControl>,
) -> Result<SearchReport, String> {
    run(config, control, None)
}

pub fn run_device(
    config: &PublicKeySearch,
    control: Arc<SearchControl>,
    device: &mut EvaluateBatch<'_>,
) -> Result<SearchReport, String> {
    run(config, control, Some(device))
}

struct Prepared<'a> {
    config: &'a PublicKeySearch,
    pattern: HexPattern,
    seed: Zeroizing<[u8; 32]>,
    deriver: CandidateDeriver,
}

impl<'a> Prepared<'a> {
    fn new(config: &'a PublicKeySearch) -> Result<Self, String> {
        config.validate()?;
        let pattern = config.pattern()?;
        let mut seed = Zeroizing::new([0; 32]);
        OsRng
            .try_fill_bytes(seed.as_mut())
            .map_err(|_| "OS cryptographic entropy unavailable")?;
        let deriver =
            CandidateDeriver::new(*seed, CandidateDomain::P256PrivateKey, [0; 32], [0; 32]);

        Ok(Self {
            config,
            pattern,
            seed,
            deriver,
        })
    }
}

fn run(
    config: &PublicKeySearch,
    control: Arc<SearchControl>,
    device: Option<&mut EvaluateBatch<'_>>,
) -> Result<SearchReport, String> {
    let prepared = Prepared::new(config)?;
    let output = match device {
        Some(device) => prepared.search_device(&control, device)?,
        None => prepared
            .search_cpu(&control)?
            .map(|winner| prepared.format_winner(winner))
            .transpose()?,
    };
    let found = output.is_some();
    if let Some(record) = &output {
        println!("{record}");
    }
    let (candidates_tested, elapsed) = control.statistics();
    Ok(SearchReport {
        candidates_tested,
        elapsed,
        found,
        output,
    })
}

pub type EvaluateBatch<'a> = dyn FnMut(
        &logic::modes::p256_public_key::P256PublicRequest,
        &logic::search::hex_pattern::HexPattern,
        &[u8],
        u64,
        u32,
    ) -> Result<logic::search::candidate_result::BatchResult, String>
    + 'a;
