//! rsa-modulus CLI entry points for the CPU and CUDA runners.
use crate::common::search_session::{RunResult, run_controlled};

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;
    use crate::common::search_session::estimate;
    pub fn run(
        args: &crate::crypto_args::RsaModulusArgs,
        workers: usize,
        stats: std::sync::Arc<crate::common::GlobalStats>,
    ) -> RunResult {
        let config = args.config(workers)?;
        estimate(config.validate()?.pattern.constrained_bits() - 2);
        println!(
            "Constructive search restricts every q candidate to the requested modulus pattern."
        );
        run_controlled(stats, "q candidates", |control| {
            vanity_miner::rsa_modulus::run_cpu(&config, control).map(|report| report.found)
        })
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    pub fn run(
        args: &crate::crypto_args::RsaModulusArgs,
        gpu: &crate::common::GpuContext,
        stats: std::sync::Arc<crate::common::GlobalStats>,
        control: std::sync::Arc<vanity_miner::search_control::SearchControl>,
    ) -> RunResult {
        let mut engine = crate::runner::cuda_batches::Engine::new(gpu);
        let config = args.config(1)?;
        {
            stats.set_unit("q candidates");
            vanity_miner::rsa_modulus::run_device(&config, control, &mut |r, p, m, s, c| {
                engine.rsa_modulus(r, p, m, s, c)
            })
            .map(|_| ())
            .map_err(Into::into)
        }
    }
}
