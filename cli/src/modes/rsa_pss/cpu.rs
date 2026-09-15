use crate::common::search_session::{RunResult, run_controlled};

use crate::common::search_session::estimate;
pub fn run(
    args: &crate::modes::rsa_pss::args::RsaPssArgs,
    workers: usize,
    stats: std::sync::Arc<crate::common::GlobalStats>,
) -> RunResult {
    let config = args.config(workers)?;
    estimate(config.validate()?.constrained_bits());
    let unit = if matches!(
        config.source,
        vanity_miner::search::rsa_pss::PssSource::Salt { .. }
    ) {
        "salts"
    } else {
        "messages"
    };
    run_controlled(stats, unit, |control| {
        vanity_miner::search::rsa_pss::run_cpu(&config, control).map(|report| report.found)
    })
}
