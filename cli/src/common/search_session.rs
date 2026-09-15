use std::{error::Error, sync::Arc};
use vanity_miner::search_control::SearchControl;

pub(crate) type RunResult = Result<(), Box<dyn Error + Send + Sync>>;

pub(crate) fn run_controlled(
    stats: Arc<crate::common::GlobalStats>,
    unit: &'static str,
    mut work: impl FnMut(Arc<SearchControl>) -> Result<bool, String>,
) -> RunResult {
    stats.set_unit(unit);
    let control = Arc::new(SearchControl::with_stats(stats.clone()));
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

#[cfg(not(feature = "gpu"))]
pub(crate) fn estimate(bits: u32) {
    println!(
        "Approximate generic random-candidate work: 16^{:.2}",
        bits as f64 / 4.0
    );
}
