use super::*;

impl Prepared<'_> {
    pub(super) fn search_cpu(&self, control: &SearchControl) -> Result<Option<Winner>, String> {
        let config = self.config;
        let pattern = &self.pattern;
        let key = &self.key;
        let public = &self.public;
        let original = &self.original;
        let digest = self.digest;
        let base_salt = &self.base_salt;
        let limit = self.limit;
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..config.workers {
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
            crate::runner::workers::join(handles, control, "RSA-PSS worker panicked")
        })
    }
}

#[cfg(not(feature = "gpu"))]
pub fn run(
    args: &crate::modes::rsa_pss::args::RsaPssArgs,
    workers: usize,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
) -> crate::runner::RunResult {
    use crate::runner::{progress::estimate, session::run_controlled};
    let config = args.config(workers)?;
    estimate(config.validate()?.constrained_bits());
    let unit = if matches!(config.source, crate::modes::rsa_pss::PssSource::Salt { .. }) {
        "salts"
    } else {
        "messages"
    };
    run_controlled(stats, unit, |control| {
        crate::modes::rsa_pss::run_cpu(&config, control).map(|report| report.found)
    })
}
