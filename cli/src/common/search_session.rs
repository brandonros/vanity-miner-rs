use std::{error::Error, sync::Arc};
use vanity_miner::search_control::SearchControl;

pub(crate) type RunResult = Result<(), Box<dyn Error + Send + Sync>>;

#[cfg(not(feature = "cumetal"))]
pub(crate) fn run_controlled(
    stats: Arc<crate::common::GlobalStats>,
    unit: &'static str,
    work: impl FnMut(Arc<SearchControl>) -> Result<bool, String>,
) -> RunResult {
    run_with_launch_limit(stats, unit, None, work)
}

pub(crate) fn run_with_launch_limit(
    stats: Arc<crate::common::GlobalStats>,
    unit: &'static str,
    launch_limit: Option<u64>,
    mut work: impl FnMut(Arc<SearchControl>) -> Result<bool, String>,
) -> RunResult {
    stats.set_unit(unit);
    let control = Arc::new(SearchControl::with_stats(stats.clone()));
    control.set_device_launch_limit(launch_limit);
    let cancellation = control.clone();
    ctrlc::set_handler(move || cancellation.interrupt())
        .map_err(|_| "could not install Ctrl-C handler")?;
    let result = (|| -> Result<(), String> {
        while !control.stopped() {
            if !work(control.clone())? {
                break;
            }
            stats.add_matches(1);
            if !control.resume_after_match() {
                break;
            }
        }
        Ok(())
    })();
    result?;
    Ok(())
}

#[cfg(not(any(feature = "gpu", feature = "cumetal")))]
pub(crate) fn estimate(bits: u32) {
    println!(
        "Approximate generic random-candidate work: 16^{:.2}",
        bits as f64 / 4.0
    );
}
