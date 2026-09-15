use crate::common::{GlobalStats, search_session::run_with_launch_limit};
use crate::runner::cumetal::{
    CumetalRunner, Error, batch_transport::CumetalBatchTransport, driver::Driver,
};
use std::{rc::Rc, sync::Arc};

pub fn run(
    runner: &CumetalRunner,
    args: &super::args::RsaPssArgs,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let config = args.config(1)?;
    let module = runner.module(driver, "kernel_rsa_pss_signature_vanity")?;
    let mut engine = CumetalBatchTransport {
        driver,
        module,
        verify: runner.options.verify,
    };
    run_with_launch_limit(
        stats,
        if matches!(
            config.source,
            vanity_miner::search::rsa_pss::PssSource::Salt { .. }
        ) {
            "salts"
        } else {
            "messages"
        },
        runner.options.batches,
        |control| {
            let report = vanity_miner::search::rsa_pss::run_device(
                &config,
                control,
                &mut |r, p, m, start, count| {
                    engine.evaluate(r, p, m, start, count, |counter| {
                        logic::modes::rsa_pss_signature_vanity::rsa_pss(r, m, counter, p)
                    })
                },
            );
            report.map(|r| r.found)
        },
    )
}
