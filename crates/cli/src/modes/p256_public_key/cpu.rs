use super::*;

impl Prepared<'_> {
    pub(super) fn search_cpu(&self, control: &SearchControl) -> Result<Option<Winner>, String> {
        let config = self.config;
        let pattern = &self.pattern;
        let deriver = &self.deriver;
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for worker in 0..config.workers {
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
            crate::runner::workers::join(handles, control, "P-256 search worker panicked")
        })
    }
}

#[cfg(not(feature = "metal"))]
pub fn run(
    args: &crate::modes::p256_public_key::args::P256PublicArgs,
    workers: usize,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
    exit_on_first_match: bool,
) -> crate::runner::RunResult {
    use crate::runner::{progress::estimate, session::run_controlled};
    let config = args.config(workers);
    let structural = if config.target == logic::crypto::p256::PublicTarget::Uncompressed {
        8
    } else {
        0
    };
    estimate(config.pattern()?.constrained_bits() - structural);
    run_controlled(stats, "keys", exit_on_first_match, |control| {
        crate::modes::p256_public_key::run_cpu(&config, control).map(|report| report.found)
    })
}
