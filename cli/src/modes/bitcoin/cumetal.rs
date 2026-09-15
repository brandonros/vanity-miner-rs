use crate::runner::{
    cumetal::{CumetalRunner, Error, batch_transport::CumetalBatchTransport, driver::Driver},
    progress::GlobalStats,
    session::run_device_session,
};
use std::{rc::Rc, sync::Arc};

pub fn run(
    runner: &CumetalRunner,
    prefix: &str,
    suffix: &str,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let module = runner.module(driver, super::device::ENTRY)?;
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
            super::device::search(
                prefix,
                suffix,
                runner.options.seed,
                &control,
                |r, p, m, start, count| {
                    engine.evaluate(r, p, m, start, count, |counter| {
                        logic::modes::bitcoin::candidate(r, counter, p)
                    })
                },
            )
            .map(|_| ())
        },
    )
}
