use crate::common::search_session::{RunResult, run_controlled};

use crate::common::search_session::estimate;
pub fn run(
    args: &crate::modes::p256_signature::args::P256SignatureArgs,
    workers: usize,
    stats: std::sync::Arc<crate::common::GlobalStats>,
) -> RunResult {
    let config = args.config(workers)?;
    estimate(config.pattern()?.constrained_bits());
    let unit = if matches!(
        config.source,
        vanity_miner::search::p256_signature::SearchSource::Message { .. }
    ) {
        "messages"
    } else {
        "nonces"
    };
    run_controlled(stats, unit, |control| {
        vanity_miner::search::p256_signature::run_cpu(&config, control).map(|report| report.found)
    })
}
