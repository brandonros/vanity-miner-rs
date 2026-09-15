use crate::runner::cumetal::{
    CumetalRunner, Error, batch_transport::CumetalBatchTransport, driver::Driver,
};
use crate::runner::{progress::GlobalStats, session::run_device_session};
use std::{rc::Rc, sync::Arc};

pub fn run(
    runner: &CumetalRunner,
    args: &super::args::P256PublicArgs,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let config = args.config(1);
    let module = runner.module(driver, "kernel_p256_public_key_vanity")?;
    let mut engine = CumetalBatchTransport::new(
        driver,
        module,
        runner.options.verify,
        runner.options.threads_per_block,
    );
    run_device_session(
        stats,
        "keys",
        runner.options.batches,
        runner.options.blocks * runner.options.threads_per_block,
        |control| {
            let report = crate::modes::p256_public_key::run_device(
                &config,
                control,
                &mut |r, p, m, start, count| {
                    engine.evaluate(r, p, m, start, count, |counter| {
                        logic::modes::p256_public_key::p256_public(r, counter, p)
                    })
                },
            );
            report.map(|_| ())
        },
    )
}
