use crate::common::search_session::RunResult;
pub fn run(
    args: &crate::modes::p256_public_key::args::P256PublicArgs,
    gpu: &crate::common::GpuContext,
    stats: std::sync::Arc<crate::common::GlobalStats>,
    control: std::sync::Arc<vanity_miner::search_control::SearchControl>,
) -> RunResult {
    let mut engine = crate::runner::cuda_transport::Engine::new(gpu);
    let config = args.config(1);
    {
        stats.set_unit("keys");
        vanity_miner::search::p256_public_key::run_device(&config, control, &mut |r, p, m, s, c| {
            engine.evaluate("kernel_p256_public_key_vanity", r, p, m, s, c)
        })
        .map(|_| ())
        .map_err(Into::into)
    }
}
