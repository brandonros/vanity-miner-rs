use crate::runner::cumetal::{
    CumetalRunner, Error, batch_transport::CumetalBatchTransport, driver::Driver,
};
use crate::runner::{progress::GlobalStats, session::run_with_launch_limit};
use std::{rc::Rc, sync::Arc};

pub fn run(
    runner: &CumetalRunner,
    args: &super::args::P256SignatureArgs,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let config = args.config(1)?;
    let module = runner.module(driver, "kernel_p256_signature_vanity")?;
    let mut engine = CumetalBatchTransport {
        driver,
        module,
        verify: runner.options.verify,
    };
    run_with_launch_limit(
        stats,
        if matches!(
            config.source,
            crate::modes::p256_signature::SearchSource::Message { .. }
        ) {
            "messages"
        } else {
            "nonces"
        },
        runner.options.batches,
        |control| {
            let report = crate::modes::p256_signature::run_device(
                &config,
                control,
                &mut |r, p, m, start, count| {
                    engine.evaluate(r, p, m, start, count, |counter| {
                        logic::modes::p256_signature::p256_signature(r, m, counter, p)
                    })
                },
            );
            report.map(|r| r.found)
        },
    )
}
