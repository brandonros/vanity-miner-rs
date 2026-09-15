use crate::common::search_session::{RunResult, run_controlled};

use crate::common::search_session::estimate;
pub fn run(
    args: &crate::modes::p256_public_key::args::P256PublicArgs,
    workers: usize,
    stats: std::sync::Arc<crate::common::GlobalStats>,
) -> RunResult {
    let config = args.config(workers);
    let structural = if config.target == logic::crypto::p256::PublicTarget::Uncompressed {
        8
    } else {
        0
    };
    estimate(config.pattern()?.constrained_bits() - structural);
    run_controlled(stats, "keys", |control| {
        vanity_miner::search::p256_public_key::run_cpu(&config, control).map(|report| report.found)
    })
}
