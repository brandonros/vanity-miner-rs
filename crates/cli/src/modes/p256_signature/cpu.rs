use super::*;

impl Prepared<'_> {
    pub(super) fn search_cpu(&self, control: &SearchControl) -> Result<Option<Winner>, String> {
        let config = self.config;
        let pattern = &self.pattern;
        let private = &self.private;
        let signing_key = &self.signing_key;
        let public = self.public;
        let original = &self.original;
        let digest = self.digest;
        let deriver = &self.deriver;
        let candidate_limit = self.candidate_limit;
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for worker in 0..config.workers {
                handles.push(scope.spawn(move || -> Result<Option<Winner>, String> {
                    let stop_peers = control.cancel_on_exit();
                    let mut message = original.clone();
                    while let Some(batch) = control.reserve_bounded_batch(64, candidate_limit) {
                        for counter in batch {
                            if control.stopped() {
                                return Ok(None);
                            }
                            let matched = match config.source {
                                SearchSource::Message { offset, length } => {
                                    write_message_counter(
                                        &mut message,
                                        offset,
                                        length,
                                        counter as u128,
                                    )
                                    .map_err(|e| e.to_string())?;
                                    let signature: Signature = signing_key
                                        .try_sign(&message)
                                        .map_err(|_| "deterministic P-256 signing failed")?;
                                    signatures::matching_representation(
                                        &signature.to_bytes().into(),
                                        config.target,
                                        config.s_form,
                                        pattern,
                                    )
                                }
                                SearchSource::Ephemeral => {
                                    let nonce =
                                        candidate_scalar(deriver, worker as u64, counter as u128)
                                            .ok_or("ephemeral scalar derivation exhausted")?;
                                    signatures::matching_ephemeral_signature(
                                        private,
                                        &digest,
                                        &nonce,
                                        config.target,
                                        config.s_form,
                                        pattern,
                                    )
                                }
                            };
                            control.add_tested(1);
                            let Some(signature) = matched else {
                                continue;
                            };
                            if !signatures::verify(&public, &message, &signature)
                                || !pattern.matches(config.target.bytes(&signature))
                            {
                                return Err("P-256 winning signature failed verification".into());
                            }
                            if control.claim_verified_winner() {
                                return Ok(Some(Winner {
                                    counter,
                                    worker: worker as u64,
                                    signature,
                                }));
                            }
                            return Ok(None);
                        }
                    }
                    stop_peers.finish();
                    Ok(None)
                }));
            }
            crate::runner::workers::join(handles, control, "P-256 signature worker panicked")
        })
    }
}

#[cfg(not(any(feature = "gpu", feature = "cumetal", feature = "metal")))]
pub fn run(
    args: &crate::modes::p256_signature::args::P256SignatureArgs,
    workers: usize,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
    exit_on_first_match: bool,
) -> crate::runner::RunResult {
    use crate::runner::{progress::estimate, session::run_controlled};
    let config = args.config(workers)?;
    estimate(config.pattern()?.constrained_bits());
    let unit = if matches!(
        config.source,
        crate::modes::p256_signature::SearchSource::Message { .. }
    ) {
        "messages"
    } else {
        "nonces"
    };
    run_controlled(stats, unit, exit_on_first_match, |control| {
        crate::modes::p256_signature::run_cpu(&config, control).map(|report| report.found)
    })
}
