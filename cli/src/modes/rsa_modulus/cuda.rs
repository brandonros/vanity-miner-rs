use crate::common::search_session::RunResult;
pub fn run(
    args: &crate::modes::rsa_modulus::args::RsaModulusArgs,
    gpu: &crate::common::GpuContext,
    stats: std::sync::Arc<crate::common::GlobalStats>,
    control: std::sync::Arc<vanity_miner::search_control::SearchControl>,
) -> RunResult {
    let mut engine = crate::runner::cuda_transport::Engine::new(gpu);
    let config = args.config(1)?;
    {
        stats.set_unit("factor candidates (p + q)");
        vanity_miner::search::rsa_modulus::run_device(&config, control, &mut |r, p, m, s, c| {
            engine.evaluate("kernel_rsa_modulus_vanity_v2", r, p, m, s, c)
        })
        .map(|_| ())
        .map_err(Into::into)
    }
}
