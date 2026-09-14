//! p256-signature CLI entry points for the CPU and CUDA runners.
use crate::common::search_session::{RunResult, run_controlled};

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;
    use crate::common::search_session::estimate;
    pub fn run(
        args: &crate::crypto_args::P256SignatureArgs,
        workers: usize,
        stats: std::sync::Arc<crate::common::GlobalStats>,
    ) -> RunResult {
        let config = args.config(workers)?;
        estimate(config.pattern()?.constrained_bits());
        let unit = if matches!(
            config.source,
            vanity_miner::p256_signature::SearchSource::Message { .. }
        ) {
            "messages tested"
        } else {
            "nonces tested"
        };
        run_controlled(stats, unit, |control| {
            vanity_miner::p256_signature::run_cpu(&config, control).map(|report| report.found)
        })
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    pub fn run(
        args: &crate::crypto_args::P256SignatureArgs,
        gpu: &crate::common::GpuContext,
        stats: std::sync::Arc<crate::common::GlobalStats>,
        control: std::sync::Arc<vanity_miner::search_control::SearchControl>,
    ) -> RunResult {
        let mut engine = crate::runner::cuda_batches::Engine::new(gpu);
        let config = args.config(1)?;
        let unit = if matches!(
            config.source,
            vanity_miner::p256_signature::SearchSource::Message { .. }
        ) {
            "messages tested"
        } else {
            "nonces tested"
        };
        {
            stats.set_unit(unit);
            vanity_miner::p256_signature::run_device(&config, control, &mut |r, p, m, s, c| {
                engine.p256_signature(r, p, m, s, c)
            })
            .map(|_| ())
            .map_err(Into::into)
        }
    }
}
