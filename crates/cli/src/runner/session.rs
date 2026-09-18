//! Coordination shared by host CPU workers and GPU device workers.

use crate::runner::progress::GlobalStats;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

const RUNNING: u8 = 0;
const CANCELLED: u8 = 1;
const WINNER: u8 = 2;

pub struct SearchControl {
    state: AtomicU8,
    interrupted: AtomicBool,
    next: AtomicU64,
    batch_size: AtomicU32,
    device_launches_remaining: AtomicU64,
    continuous: AtomicBool,
    exit_on_first_match: AtomicBool,
    stats: Arc<GlobalStats>,
}

impl Default for SearchControl {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchControl {
    pub fn new() -> Self {
        Self::with_stats(Arc::new(GlobalStats::new(1, 0, 0)))
    }

    pub fn with_stats(stats: Arc<GlobalStats>) -> Self {
        Self {
            state: AtomicU8::new(RUNNING),
            interrupted: AtomicBool::new(false),
            next: AtomicU64::new(0),
            batch_size: AtomicU32::new(64),
            device_launches_remaining: AtomicU64::new(u64::MAX),
            continuous: AtomicBool::new(false),
            exit_on_first_match: AtomicBool::new(false),
            stats,
        }
    }

    /// Candidate count per device launch; finite searches clamp the final batch.
    pub fn set_batch_size(&self, count: u32) -> Result<(), &'static str> {
        if count == 0 || count > 1_048_576 {
            return Err("batch size must be between 1 and 1048576");
        }
        self.batch_size.store(count, Ordering::Relaxed);
        Ok(())
    }

    /// Configure a session-wide launch limit before starting its workers.
    pub fn set_device_launch_limit(&self, limit: Option<u64>) {
        self.device_launches_remaining
            .store(limit.unwrap_or(u64::MAX), Ordering::Release);
    }

    /// Consume one launch permit. Exhaustion stops before another launch and
    /// leaves the previous batch available for winner verification and output.
    pub fn reserve_device_launch(&self) -> bool {
        if self.stopped() {
            return false;
        }
        let reserved = self
            .device_launches_remaining
            .fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |remaining| match remaining {
                    u64::MAX => Some(u64::MAX),
                    0 => None,
                    n => Some(n - 1),
                },
            )
            .is_ok();
        if !reserved {
            self.cancel();
        }
        reserved
    }

    pub fn batch_size(&self) -> u32 {
        self.batch_size.load(Ordering::Relaxed)
    }

    /// Configure once before starting independent device searches.
    pub fn set_continuous(&self) {
        if !self.exit_on_first_match() {
            self.continuous.store(true, Ordering::Release);
        }
    }

    /// Configure before workers start. Winner verification completes before the
    /// atomic claim; the winner remains stopped after its output is printed.
    pub fn set_exit_on_first_match(&self) {
        self.exit_on_first_match.store(true, Ordering::Release);
        self.continuous.store(false, Ordering::Release);
    }

    pub fn exit_on_first_match(&self) -> bool {
        self.exit_on_first_match.load(Ordering::Acquire)
    }

    pub fn continuous(&self) -> bool {
        self.continuous.load(Ordering::Acquire)
    }

    pub fn add_verified_match(&self) {
        self.stats.add_matches(1);
    }

    pub fn has_winner(&self) -> bool {
        self.state.load(Ordering::Acquire) == WINNER
    }

    pub fn stopped(&self) -> bool {
        self.interrupted.load(Ordering::Acquire) || self.state.load(Ordering::Acquire) != RUNNING
    }

    /// Process/user cancellation persists across match boundaries.
    pub fn interrupt(&self) {
        self.interrupted.store(true, Ordering::Release);
        self.cancel();
    }

    /// Called after every worker has joined and the winning record was printed.
    /// Keep the counter advancing so message searches never repeat candidates.
    pub fn resume_after_match(&self) -> bool {
        !self.exit_on_first_match()
            && !self.interrupted.load(Ordering::Acquire)
            && self
                .state
                .compare_exchange(WINNER, RUNNING, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
    }

    pub fn cancel(&self) {
        let _ =
            self.state
                .compare_exchange(RUNNING, CANCELLED, Ordering::AcqRel, Ordering::Acquire);
    }

    /// Reserve disjoint global candidate identifiers. Overflow ends the search;
    /// the same identifiers must never be reused with the same secret seed.
    pub fn reserve_batch(&self, size: u64) -> Option<Range<u64>> {
        if size == 0 || self.stopped() {
            return None;
        }
        match self
            .next
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |next| {
                next.checked_add(size)
            }) {
            Ok(start) => Some(start..start + size),
            Err(_) => {
                self.cancel();
                None
            }
        }
    }

    /// Call only after complete winner verification. Claiming stops all workers,
    /// until the host has printed the match and joined the workers.
    pub fn claim_verified_winner(&self) -> bool {
        self.state
            .compare_exchange(RUNNING, WINNER, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Reserve up to the remaining finite candidate space. Exhaustion does not
    /// cancel peers: their already-reserved batches must finish first.
    pub fn reserve_bounded_batch(&self, size: u64, end: u64) -> Option<Range<u64>> {
        if size == 0 || self.stopped() {
            return None;
        }
        self.next
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |next| {
                (next < end).then(|| next.saturating_add(size).min(end))
            })
            .ok()
            .map(|start| start..start.saturating_add(size).min(end))
    }

    /// Any worker exit stops peers immediately, including an error returned
    /// before the joining thread reaches that worker's handle.
    pub fn cancel_on_exit(&self) -> CancelOnExit<'_> {
        CancelOnExit(Some(self))
    }

    pub fn add_tested(&self, count: u64) {
        self.stats.add_operations(count);
    }

    pub fn statistics(&self) -> (u64, Duration) {
        self.stats.statistics()
    }
}

pub struct CancelOnExit<'a>(Option<&'a SearchControl>);
impl CancelOnExit<'_> {
    /// A worker that completed a finite partition must let its peers finish.
    pub fn finish(mut self) {
        self.0 = None;
    }
}
impl Drop for CancelOnExit<'_> {
    fn drop(&mut self) {
        if let Some(control) = self.0 {
            control.cancel();
        }
    }
}

#[cfg(all(feature = "crypto-cli", not(feature = "metal")))]
use crate::runner::RunResult;

#[cfg(all(feature = "crypto-cli", not(feature = "metal")))]
pub(crate) fn run_controlled(
    stats: Arc<crate::runner::progress::GlobalStats>,
    unit: &'static str,
    exit_on_first_match: bool,
    mut work: impl FnMut(Arc<SearchControl>) -> Result<bool, String>,
) -> RunResult {
    stats.set_unit(unit);
    let control = Arc::new(SearchControl::with_stats(stats.clone()));
    if exit_on_first_match {
        control.set_exit_on_first_match();
    }
    let cancellation = control.clone();
    ctrlc::set_handler(move || cancellation.interrupt())
        .map_err(|_| "could not install Ctrl-C handler")?;
    let result = (|| -> Result<(), String> {
        while !control.stopped() {
            if !work(control.clone())? {
                break;
            }
            if !control.resume_after_match() {
                break;
            }
        }
        Ok(())
    })();
    result?;
    Ok(())
}

/// Run a continuous device session once, preserving counters and prepared state.
#[cfg(feature = "metal")]
pub(crate) fn run_device_session(
    stats: Arc<GlobalStats>,
    unit: &'static str,
    launches: Option<u64>,
    batch_size: u32,
    exit_on_first_match: bool,
    work: impl FnOnce(Arc<SearchControl>) -> Result<(), String>,
) -> crate::runner::RunResult {
    stats.set_unit(unit);
    let control = Arc::new(SearchControl::with_stats(stats));
    if exit_on_first_match {
        control.set_exit_on_first_match();
    }
    control.set_batch_size(batch_size)?;
    control.set_device_launch_limit(launches);
    control.set_continuous();
    let cancellation = control.clone();
    ctrlc::set_handler(move || cancellation.interrupt()).map_err(|e| e.to_string())?;
    work(control).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn first_match_policy_prevents_continuous_restart() {
        let control = SearchControl::new();
        control.set_continuous();
        control.set_exit_on_first_match();
        control.set_continuous();
        assert!(!control.continuous());
        assert!(control.claim_verified_winner());
        assert!(!control.resume_after_match());
        assert!(control.stopped());
        assert!(control.reserve_batch(1).is_none());
    }

    #[test]
    fn only_one_verified_worker_wins() {
        let control = Arc::new(SearchControl::new());
        let barrier = Arc::new(Barrier::new(16));
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let control = control.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    control.claim_verified_winner()
                })
            })
            .collect();
        let winners = handles
            .into_iter()
            .filter_map(|handle| handle.join().unwrap().then_some(()))
            .count();
        assert_eq!(winners, 1);
        assert!(control.stopped());
        assert!(control.reserve_batch(1).is_none());
    }

    #[test]
    fn batches_do_not_overlap() {
        let control = Arc::new(SearchControl::new());
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let control = control.clone();
                std::thread::spawn(move || control.reserve_batch(32).unwrap().collect::<Vec<_>>())
            })
            .collect();
        let mut ids: Vec<_> = handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..256).collect::<Vec<_>>());
        assert!(control.reserve_batch(0).is_none());
    }

    #[test]
    fn finite_batches_finish_without_cancelling_peers() {
        let control = SearchControl::new();
        assert_eq!(control.reserve_bounded_batch(64, 65), Some(0..64));
        assert_eq!(control.reserve_bounded_batch(64, 65), Some(64..65));
        assert_eq!(control.reserve_bounded_batch(64, 65), None);
        control.cancel_on_exit().finish();
        assert!(!control.stopped());
        assert!(control.claim_verified_winner());
    }

    #[test]
    fn worker_exit_cancels_peers() {
        let control = SearchControl::new();
        {
            let _guard = control.cancel_on_exit();
        }
        assert!(control.stopped());
        assert!(!control.claim_verified_winner());
    }

    #[test]
    fn cancellation_and_exhaustion_do_not_restart() {
        let control = SearchControl::new();
        control.next.store(u64::MAX - 2, Ordering::Relaxed);
        assert_eq!(control.reserve_batch(2), Some(u64::MAX - 2..u64::MAX));
        assert!(control.reserve_batch(1).is_none());
        assert!(control.stopped());
        assert!(!control.claim_verified_winner());
        let cancelled = SearchControl::new();
        cancelled.cancel();
        assert!(!cancelled.claim_verified_winner());
        assert!(cancelled.reserve_batch(1).is_none());
    }
}

#[cfg(test)]
mod continuous_tests {
    use super::*;

    #[test]
    fn resuming_after_match_keeps_candidate_counters() {
        let control = SearchControl::new();
        assert_eq!(control.reserve_bounded_batch(64, 256), Some(0..64));
        assert!(control.claim_verified_winner());
        assert!(control.resume_after_match());
        assert_eq!(control.reserve_bounded_batch(64, 256), Some(64..128));
        assert!(control.claim_verified_winner());
        assert!(control.resume_after_match());
        assert_eq!(control.reserve_bounded_batch(64, 256), Some(128..192));
    }

    #[test]
    fn ctrl_c_during_winner_output_prevents_restart() {
        let control = SearchControl::new();
        assert!(control.claim_verified_winner());
        control.interrupt();
        assert!(!control.resume_after_match());
        assert!(control.stopped());
        assert!(control.reserve_batch(64).is_none());
    }
}
