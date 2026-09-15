use crate::common::search_session::RunResult;
pub fn run(
    args: &crate::modes::p256_signature::args::P256SignatureArgs,
    gpu: &crate::common::GpuContext,
    stats: std::sync::Arc<crate::common::GlobalStats>,
    control: std::sync::Arc<vanity_miner::search_control::SearchControl>,
) -> RunResult {
    let mut engine = crate::runner::cuda_transport::Engine::new(gpu);
    let config = args.config(1)?;
    let unit = if matches!(
        config.source,
        vanity_miner::search::p256_signature::SearchSource::Message { .. }
    ) {
        "messages"
    } else {
        "nonces"
    };
    {
        stats.set_unit(unit);
        vanity_miner::search::p256_signature::run_device(&config, control, &mut |r, p, m, s, c| {
            engine.evaluate("kernel_p256_signature_vanity", r, p, m, s, c)
        })
        .map(|_| ())
        .map_err(Into::into)
    }
}
