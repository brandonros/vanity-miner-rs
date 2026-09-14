use std::{
    error::Error,
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};
use vanity_miner::search_control::SearchControl;

pub(crate) type RunResult = Result<(), Box<dyn Error + Send + Sync>>;

pub(crate) fn run_controlled(
    stats: Arc<crate::common::GlobalStats>,
    unit: &'static str,
    work: impl FnOnce(Arc<SearchControl>) -> Result<bool, String>,
) -> RunResult {
    stats.set_unit(unit);
    let control = Arc::new(SearchControl::with_stats(stats.clone()));
    let cancellation = control.clone();
    ctrlc::set_handler(move || cancellation.cancel())
        .map_err(|_| "could not install Ctrl-C handler")?;
    let observed = stats.clone();
    let (done, finished) = mpsc::channel();
    let monitor = thread::spawn(move || {
        while matches!(
            finished.recv_timeout(Duration::from_secs(2)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ) {
            observed.print_progress();
        }
    });
    let result = work(control.clone());
    let _ = done.send(());
    monitor.join().map_err(|_| "statistics worker failed")?;
    stats.print_progress();
    if result? {
        stats.add_matches(1);
        println!("Verified match written.");
    } else {
        println!("Search cancelled.");
    }
    Ok(())
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn estimate(bits: u32) {
    println!(
        "Approximate generic random-candidate work: 16^{:.2}",
        bits as f64 / 4.0
    );
}
