use super::shared_best_hash::SharedBestHash;
use crate::runner::{
    cumetal::{CumetalRunner, Error, batch_transport::CumetalBatchTransport, driver::Driver},
    progress::GlobalStats,
    session::run_device_session,
};
use std::{
    rc::Rc,
    sync::{Arc, RwLock},
};

pub fn run(
    runner: &CumetalRunner,
    username: &str,
    target_hash: &str,
    driver: &Rc<Driver>,
    stats: Arc<GlobalStats>,
) -> Result<(), Error> {
    let best = Arc::new(RwLock::new(SharedBestHash::new(
        hex::decode(target_hash)?
            .try_into()
            .map_err(|_| "invalid target width")?,
    )));
    let module = runner.module(driver, super::device::ENTRY)?;
    let mut engine = CumetalBatchTransport::new(
        driver,
        module,
        runner.options.verify,
        runner.options.threads_per_block,
    );
    run_device_session(
        stats,
        "nonces",
        runner.options.batches,
        runner.options.blocks * runner.options.threads_per_block,
        |control| {
            super::device::search(
                username,
                best,
                runner.options.seed,
                &control,
                |r, p, m, start, count| {
                    engine.evaluate(r, p, m, start, count, |counter| {
                        logic::modes::shallenge::candidate(r, counter, p, m)
                    })
                },
            )
            .map(|_| ())
        },
    )
}
