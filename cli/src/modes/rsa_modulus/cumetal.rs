use crate::runner::cumetal::{
    CumetalRunner, Error, batch_transport::CumetalBatchTransport, driver::Driver,
};
use crate::runner::{progress::GlobalStats, session::run_with_launch_limit};
use std::{rc::Rc, sync::Arc};

pub fn run(
    runner: &CumetalRunner,
    args: &super::args::RsaModulusArgs,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let config = args.config(1)?;
    let module = runner.module(driver, "kernel_rsa_modulus_vanity_v2")?;
    let mut engine = CumetalBatchTransport {
        driver,
        module,
        verify: runner.options.verify,
    };
    run_with_launch_limit(
        stats,
        "factor candidates (p + q)",
        runner.options.batches,
        |control| {
            let report = crate::modes::rsa_modulus::run_device(
                &config,
                control,
                &mut |r, p, m, start, count| {
                    engine.evaluate(r, p, m, start, count, |counter| {
                        logic::modes::rsa_modulus::rsa_modulus(r, counter, p)
                    })
                },
            );
            report.map(|r| r.found)
        },
    )
}
