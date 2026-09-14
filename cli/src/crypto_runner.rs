use std::{
    error::Error,
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};
use vanity_miner::search_control::SearchControl;

type RunResult = Result<(), Box<dyn Error + Send + Sync>>;

fn statistics(control: &SearchControl, unit: &str) {
    let (tested, elapsed) = control.statistics();
    let seconds = elapsed.as_secs_f64();
    println!(
        "{tested} {unit} in {seconds:.2}s ({:.2}/s)",
        tested as f64 / seconds.max(1e-9)
    );
}

pub(crate) fn run_controlled(
    unit: &'static str,
    work: impl FnOnce(Arc<SearchControl>) -> Result<bool, String>,
) -> RunResult {
    let control = Arc::new(SearchControl::new());
    let cancellation = control.clone();
    ctrlc::set_handler(move || cancellation.cancel())
        .map_err(|_| "could not install Ctrl-C handler")?;
    let observed = control.clone();
    let (done, finished) = mpsc::channel();
    let monitor = thread::spawn(move || {
        while matches!(
            finished.recv_timeout(Duration::from_secs(2)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ) {
            statistics(&observed, unit);
        }
    });
    let result = work(control.clone());
    let _ = done.send(());
    monitor.join().map_err(|_| "statistics worker failed")?;
    statistics(&control, unit);
    if result? {
        println!("Verified match written.");
    } else {
        println!("Search cancelled.");
    }
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn estimate(bits: u32) {
    println!(
        "Approximate generic random-candidate work: 16^{:.2}",
        bits as f64 / 4.0
    );
}

#[cfg(all(feature = "rsa-modulus", not(feature = "gpu")))]
pub fn modulus(args: &crate::crypto_args::RsaModulusArgs, workers: usize) -> RunResult {
    let config = args.config(workers)?;
    estimate(config.validate()?.pattern.constrained_bits() - 2);
    println!("Constructive search restricts every q candidate to the requested modulus pattern.");
    run_controlled("q candidates tested", |control| {
        vanity_miner::rsa_modulus::run_cpu(&config, control).map(|report| report.found)
    })
}

#[cfg(all(feature = "rsa-pss", not(feature = "gpu")))]
pub fn pss(args: &crate::crypto_args::RsaPssArgs, workers: usize) -> RunResult {
    let config = args.config(workers)?;
    estimate(config.validate()?.constrained_bits());
    let unit = if matches!(
        config.source,
        vanity_miner::rsa_pss_search::PssSource::Salt { .. }
    ) {
        "salts tested"
    } else {
        "messages tested"
    };
    run_controlled(unit, |control| {
        vanity_miner::rsa_pss_search::run_cpu(&config, control).map(|report| report.found)
    })
}

#[cfg(all(feature = "p256-public-key", not(feature = "gpu")))]
pub fn public_key(args: &crate::crypto_args::P256PublicArgs, workers: usize) -> RunResult {
    let config = args.config(workers);
    let structural = if config.target == logic::p256_vanity::PublicTarget::Uncompressed {
        8
    } else {
        0
    };
    estimate(config.pattern()?.constrained_bits() - structural);
    run_controlled("keys tested", |control| {
        vanity_miner::p256_public::run_cpu(&config, control).map(|report| report.found)
    })
}

#[cfg(all(feature = "p256-signature", not(feature = "gpu")))]
pub fn signature(args: &crate::crypto_args::P256SignatureArgs, workers: usize) -> RunResult {
    let config = args.config(workers)?;
    estimate(config.pattern()?.constrained_bits());
    let unit = if matches!(
        config.source,
        vanity_miner::p256_signature::SearchSource::Message { .. }
    ) {
        "messages tested"
    } else {
        "nonces tested"
    };
    run_controlled(unit, |control| {
        vanity_miner::p256_signature::run_cpu(&config, control).map(|report| report.found)
    })
}
