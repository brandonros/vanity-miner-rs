//! Bounded host runner for P-256 public-key searches.

use crate::search_control::SearchControl;
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

/// Host reconstruction: validate SEC1/on-curve encoding, derive the public point
/// again from the scalar, and check both complete byte equality and the pattern.
fn verify_winner(winner: &Winner, target: PublicTarget, pattern: &HexPattern) -> bool {
    use p256::elliptic_curve::sec1::ToEncodedPoint;
    let Ok(private) = SecretKey::from_slice(winner.private.as_ref()) else {
        return false;
    };
    let Ok(public) = p256::PublicKey::from_sec1_bytes(&winner.public) else {
        return false;
    };
    // P-256 has cofactor one; a valid nonidentity public point has prime order.
    let derived = private.public_key();
    derived == public
        && derived.to_encoded_point(false).as_bytes() == winner.public
        && pattern.matches(target.bytes(&winner.public))
}

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

fn run(
    config: &PublicKeySearch,
    control: Arc<SearchControl>,
    device: Option<&mut EvaluateBatch<'_>>,
) -> Result<SearchReport, String> {
    config.validate()?;
    let pattern = config.pattern()?;
    let mut seed = Zeroizing::new([0; 32]);
    OsRng
        .try_fill_bytes(seed.as_mut())
        .map_err(|_| "OS cryptographic entropy unavailable")?;
    let deriver = CandidateDeriver::new(*seed, CandidateDomain::P256PrivateKey, [0; 32], [0; 32]);

    let outcome = if let Some(device) = device {
        use logic::modes::p256_public_key_vanity::P256PublicRequest;
        let request = Zeroizing::new(P256PublicRequest {
            seed: *seed,
            worker: 0,
            target: match config.target {
                PublicTarget::X => 0,
                PublicTarget::Y => 1,
                PublicTarget::Xy => 2,
                PublicTarget::Uncompressed => 3,
            },
            reserved: 0,
        });
        let mut winner = None;
        let found = crate::search_batches::find(
            |start, count| device(&request, &pattern, &[], start, count),
            u64::MAX,
            &control,
            |counter, bytes| {
                let private = candidate_scalar(&deriver, 0, counter as u128)
                    .ok_or("device scalar reconstruction failed")?;
                let public = bytes[..65]
                    .try_into()
                    .map_err(|_| "invalid device point width")?;
                let candidate = Winner { private, public };
                if !verify_winner(&candidate, config.target, &pattern) {
                    return Err("device P-256 public key failed verification".into());
                }
                winner = Some(candidate);
                Ok(true)
            },
        )?;
        if found.is_some() { winner } else { None }
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for worker in 0..config.workers {
                let control = &control;
                let deriver = &deriver;
                let pattern = &pattern;
                handles.push(scope.spawn(move || -> Result<Option<Winner>, String> {
                    let _cancel_on_exit = control.cancel_on_exit();
                    while let Some(batch) = control.reserve_batch(64) {
                        for counter in batch {
                            if control.stopped() {
                                return Ok(None);
                            }
                            let private = candidate_scalar(deriver, worker as u64, counter as u128)
                                .ok_or("P-256 scalar derivation exhausted")?;
                            let public =
                                public_point(&private).ok_or("P-256 public derivation failed")?;
                            control.add_tested(1);
                            if !pattern.matches(config.target.bytes(&public)) {
                                continue;
                            }
                            let winner = Winner { private, public };
                            if !verify_winner(&winner, config.target, pattern) {
                                control.cancel();
                                return Err("P-256 winner failed host verification".into());
                            }
                            if control.claim_verified_winner() {
                                return Ok(Some(winner));
                            }
                            return Ok(None);
                        }
                    }
                    Ok(None)
                }));
            }
            super::worker_results::join(handles, &control, "P-256 search worker panicked")
        })?
    };
    let found = outcome.is_some();
    let mut output = None;
    if let Some(winner) = outcome {
        // Recheck immediately before printing the matched key.
        if !verify_winner(&winner, config.target, &pattern) {
            return Err("P-256 winner failed final verification".into());
        }
        output = Some(format!(
            "[p256-public-key] public_key={}\n[p256-public-key] sec1_public_key={}\n[p256-public-key] private_key={}",
            hex::encode(config.target.bytes(&winner.public)),
            hex::encode(winner.public),
            hex::encode(*winner.private),
        ));
    }
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
        &logic::modes::p256_public_key_vanity::P256PublicRequest,
        &logic::search::hex_pattern::HexPattern,
        &[u8],
        u64,
        u32,
    ) -> Result<logic::search::candidate_result::BatchResult, String>
    + 'a;

#[cfg(test)]
mod tests;
