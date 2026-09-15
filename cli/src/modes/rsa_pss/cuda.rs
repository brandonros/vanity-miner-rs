use crate::common::search_session::RunResult;
pub fn run(
    args: &crate::modes::rsa_pss::args::RsaPssArgs,
    gpu: &crate::common::GpuContext,
    stats: std::sync::Arc<crate::common::GlobalStats>,
    control: std::sync::Arc<vanity_miner::search_control::SearchControl>,
) -> RunResult {
    let mut engine = crate::runner::cuda_transport::CudaBatchTransport::new(gpu);
    let config = args.config(1)?;
    let unit = if matches!(
        config.source,
        vanity_miner::search::rsa_pss::PssSource::Salt { .. }
    ) {
        "salts"
    } else {
        "messages"
    };
    stats.set_unit(unit);
    vanity_miner::search::rsa_pss::run_device(&config, control, &mut |r, p, m, s, c| {
        engine.evaluate("kernel_rsa_pss_signature_vanity", r, p, m, s, c)
    })
    .map(|_| ())
    .map_err(Into::into)
}
