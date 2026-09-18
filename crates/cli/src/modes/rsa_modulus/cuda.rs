use crate::runner::RunResult;
pub fn run(
    args: &crate::modes::rsa_modulus::args::RsaModulusArgs,
    gpu: &crate::runner::cuda::context::GpuContext,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
    control: std::sync::Arc<crate::runner::session::SearchControl>,
) -> RunResult {
    let mut engine = None;
    let config = args.config(1)?;
    stats.set_unit("candidates");
    crate::modes::rsa_modulus::pipeline::run(&config, &control, |r, p, s, c| {
        if engine.is_none() {
            engine = Some(crate::runner::cuda::batch::CandidateBatch::new(
                gpu,
                logic::modes::rsa_modulus::ENTRY,
                r,
                p,
                &[],
            )?);
        }
        engine.as_mut().unwrap().evaluate(s, c)
    })
    .map_err(Into::into)
}
