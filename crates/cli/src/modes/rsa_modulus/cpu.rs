//! CPU workers invoke the same no_std mining function as the GPU wrapper.
use super::*;
use logic::modes::rsa_modulus::{self as mining, SearchConfig, Task};

pub(super) fn construct_worker(
    config: &SearchConfig,
    pattern: &HexPattern,
    control: &SearchControl,
    steps: u32,
    stages: &pipeline::StageStats,
) -> Result<Option<String>, String> {
    let _stop_peers = control.cancel_on_exit();
    let mut task = Zeroizing::new(Task::EMPTY);
    while !control.stopped() {
        let Some(ids) = control.reserve_batch(u64::from(steps)) else {
            break;
        };
        let (counts, pair) = mining::mine(config, pattern, &mut task, ids.start, 1, steps);
        counts.validate(1, steps)?;
        control.add_tested(u64::from(counts.q_tested));
        stages.add(
            u64::from(counts.p_tested),
            u64::from(counts.p_accepted),
            u64::from(counts.ranges),
            u64::from(counts.q_tested),
        );
        if let Some(pair) = pair {
            let pair = Zeroizing::new(pair);
            let output = pipeline::verify_pair(config, pattern, &pair)?;
            if control.claim_verified_winner() {
                return Ok(Some(output));
            }
            return Ok(None);
        }
    }
    Ok(None)
}

#[cfg(not(any(feature = "gpu", feature = "cumetal", feature = "metal")))]
pub fn run(
    args: &crate::modes::rsa_modulus::args::RsaModulusArgs,
    workers: usize,
    stats: std::sync::Arc<crate::runner::progress::GlobalStats>,
) -> crate::runner::RunResult {
    use crate::runner::{progress::estimate, session::run_controlled};
    let config = args.config(workers)?;
    estimate(config.validate()?.pattern.constrained_bits() - 2);
    println!("Constructive search restricts every q candidate to the requested modulus pattern.");
    // Attach once per session, retaining totals when a match restarts the workers.
    let stages = pipeline::StageStats::attach(&stats)?;
    run_controlled(stats, "q candidates", |control| {
        crate::modes::rsa_modulus::run_cpu_with_steps(
            &config,
            control,
            args.steps_per_launch,
            &stages,
        )
        .map(|report| report.found)
    })
}
