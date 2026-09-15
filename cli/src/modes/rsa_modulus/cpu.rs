use crate::common::search_session::{RunResult, run_controlled};

use crate::common::search_session::estimate;
pub fn run(
    args: &crate::modes::rsa_modulus::args::RsaModulusArgs,
    workers: usize,
    stats: std::sync::Arc<crate::common::GlobalStats>,
) -> RunResult {
    let config = args.config(workers)?;
    estimate(config.validate()?.pattern.constrained_bits() - 2);
    println!("Constructive search restricts every q candidate to the requested modulus pattern.");
    run_controlled(stats, "q candidates", |control| {
        vanity_miner::search::rsa_modulus::run_cpu(&config, control).map(|report| report.found)
    })
}
