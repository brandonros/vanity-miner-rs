use crate::common::{GlobalStats, search_session::run_with_launch_limit};
use crate::runner::cumetal::{CumetalRunner, Error, batch_transport::Engine, driver::Driver};
use std::{rc::Rc, sync::Arc};

pub fn run(
    runner: &CumetalRunner,
    args: &super::args::P256PublicArgs,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let config = args.config(1);
    let module = runner.module(driver, "kernel_p256_public_key_vanity")?;
    let mut engine = Engine {
        driver,
        module,
        verify: runner.options.verify,
    };
    run_with_launch_limit(stats, "keys", runner.options.batches, |control| {
        let report = vanity_miner::search::p256_public_key::run_device(
            &config,
            control,
            &mut |r, p, m, start, count| {
                engine.evaluate(r, p, m, start, count, |counter| {
                    logic::modes::p256_public_key_vanity::p256_public(r, counter, p)
                })
            },
        );
        report.map(|r| r.found)
    })
}
