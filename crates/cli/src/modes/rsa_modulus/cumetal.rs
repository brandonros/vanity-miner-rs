use crate::runner::cumetal::{
    CumetalRunner, Error, batch_transport::CumetalBatchTransport, driver::Driver,
};
use crate::runner::{progress::GlobalStats, session::run_device_session};
use std::{rc::Rc, sync::Arc};

pub fn run(
    runner: &CumetalRunner,
    args: &super::args::RsaModulusArgs,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let config = args.config(1)?;
    let module = runner.module(driver, logic::modes::rsa_modulus::ENTRY)?;
    let mut engine = CumetalBatchTransport::new(
        driver,
        module,
        runner.options.verify,
        runner.options.threads_per_block,
    );
    run_device_session(
        stats,
        "candidates",
        runner.options.batches,
        runner.options.blocks * runner.options.threads_per_block,
        |control| {
            super::pipeline::run(&config, &control, |r, p, start, count| {
                engine.evaluate(r, p, &[], start, count, |counter| {
                    logic::modes::rsa_modulus::rsa_modulus(r, counter, p)
                })
            })
        },
    )
}
