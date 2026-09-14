//! p256-public-key CLI entry points for the CPU and CUDA runners.
use crate::common::search_session::{RunResult, run_controlled};

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;
    use crate::common::search_session::estimate;
    pub fn run(
        args: &crate::crypto_args::P256PublicArgs,
        workers: usize,
        stats: std::sync::Arc<crate::common::GlobalStats>,
    ) -> RunResult {
        let config = args.config(workers);
        let structural = if config.target == logic::p256_vanity::PublicTarget::Uncompressed {
            8
        } else {
            0
        };
        estimate(config.pattern()?.constrained_bits() - structural);
        run_controlled(stats, "keys tested", |control| {
            vanity_miner::p256_public::run_cpu(&config, control).map(|report| report.found)
        })
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    pub fn run(
        args: &crate::crypto_args::P256PublicArgs,
        gpu: &crate::common::GpuContext,
        stats: std::sync::Arc<crate::common::GlobalStats>,
        control: std::sync::Arc<vanity_miner::search_control::SearchControl>,
    ) -> RunResult {
        let mut engine = crate::runner::cuda_batches::Engine::new(gpu);
        let config = args.config(1);
        {
            stats.set_unit("keys tested");
            vanity_miner::p256_public::run_device(&config, control, &mut |r, p, m, s, c| {
                engine.p256_public(r, p, m, s, c)
            })
            .map(|_| ())
            .map_err(Into::into)
        }
    }
}
