use crate::runner::RunResult;
pub fn run(
    args: &crate::modes::p256_public_key::args::P256PublicArgs,
    gpu: &crate::runner::cuda::context::GpuContext,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
    control: std::sync::Arc<crate::runner::session::SearchControl>,
) -> RunResult {
    let mut engine = None;
    let config = args.config(1);
    stats.set_unit("keys");
    crate::modes::p256_public_key::run_device(&config, control, &mut |r, p, m, s, c| {
        if engine.is_none() {
            engine = Some(crate::runner::cuda::batch::CandidateBatch::new(
                gpu,
                "kernel_p256_public_key_vanity",
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
