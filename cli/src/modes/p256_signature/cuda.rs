use crate::runner::RunResult;
pub fn run(
    args: &crate::modes::p256_signature::args::P256SignatureArgs,
    gpu: &crate::runner::cuda::context::GpuContext,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
    control: std::sync::Arc<crate::runner::session::SearchControl>,
) -> RunResult {
    let mut engine = None;
    let config = args.config(1)?;
    let unit = if matches!(
        config.source,
        crate::modes::p256_signature::SearchSource::Message { .. }
    ) {
        "messages"
    } else {
        "nonces"
    };
    stats.set_unit(unit);
    crate::modes::p256_signature::run_device(&config, control, &mut |r, p, m, s, c| {
        if engine.is_none() {
            engine = Some(crate::runner::cuda::batch::CandidateBatch::new(
                gpu,
                "kernel_p256_signature_vanity",
                r,
                p,
                m,
            )?);
        }
        engine.as_mut().unwrap().evaluate(s, c)
    })
    .map(|_| ())
    .map_err(Into::into)
}
