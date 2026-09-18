use crate::runner::cumetal::{
    CumetalRunner, Error, batch_transport::CumetalBatchTransport, driver::Driver,
};
use crate::runner::{progress::GlobalStats, session::run_device_session};
use std::{rc::Rc, sync::Arc};

pub fn run(
    runner: &CumetalRunner,
    args: &super::args::RsaPssArgs,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let config = args.config(1)?;
    let module = runner.module(driver, "kernel_rsa_pss_signature_vanity")?;
    let mut engine = CumetalBatchTransport::new(
        driver,
        module,
        runner.options.verify,
        runner.options.threads_per_block,
    );
    run_device_session(
        stats,
        if matches!(config.source, crate::modes::rsa_pss::PssSource::Salt { .. }) {
            "salts"
        } else {
            "messages"
        },
        runner.options.batches,
        runner.options.blocks * runner.options.threads_per_block,
        runner.exit_on_first_match,
        |control| {
            let report = crate::modes::rsa_pss::run_device(
                &config,
                control,
                &mut |r, p, m, start, count| {
                    engine.evaluate(r, p, m, start, count, |counter| {
                        logic::modes::rsa_pss::rsa_pss(r, m, counter, p)
                    })
                },
            );
            report.map(|_| ())
        },
    )
}
