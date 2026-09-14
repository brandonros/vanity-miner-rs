//! rsa-pss CLI entry points for the CPU and CUDA runners.
use crate::common::search_session::{RunResult, run_controlled};

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;
    use crate::common::search_session::estimate;
    pub fn run(
        args: &crate::crypto_args::RsaPssArgs,
        workers: usize,
        stats: std::sync::Arc<crate::common::GlobalStats>,
    ) -> RunResult {
        let config = args.config(workers)?;
        estimate(config.validate()?.constrained_bits());
        let unit = if matches!(
            config.source,
            vanity_miner::rsa_pss_search::PssSource::Salt { .. }
        ) {
            "salts tested"
        } else {
            "messages tested"
        };
        run_controlled(stats, unit, |control| {
            vanity_miner::rsa_pss_search::run_cpu(&config, control).map(|report| report.found)
        })
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    pub fn run(
        args: &crate::crypto_args::RsaPssArgs,
        gpu: &crate::common::GpuContext,
        stats: std::sync::Arc<crate::common::GlobalStats>,
        control: std::sync::Arc<vanity_miner::search_control::SearchControl>,
    ) -> RunResult {
        let mut engine = crate::runner::cuda_batches::Engine::new(gpu);
        let config = args.config(1)?;
        let unit = if matches!(
            config.source,
            vanity_miner::rsa_pss_search::PssSource::Salt { .. }
        ) {
            "salts tested"
        } else {
            "messages tested"
        };
        {
            stats.set_unit(unit);
            vanity_miner::rsa_pss_search::run_device(&config, control, &mut |r, p, m, s, c| {
                engine.rsa_pss(r, p, m, s, c)
            })
            .map(|_| ())
            .map_err(Into::into)
        }
    }
}
