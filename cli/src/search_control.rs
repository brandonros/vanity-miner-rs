//! Coordination shared by host CPU workers and GPU device workers.

use std::ops::Range;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub struct SearchControl {
    // 0 = running, 1 = cancelled, 2 = verified winner reserved.
    state: AtomicU8,
    next: AtomicU64,
    tested: AtomicU64,
    start: Instant,
}

impl Default for SearchControl {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchControl {
    pub fn new() -> Self {
        Self {
            state: AtomicU8::new(0),
            next: AtomicU64::new(0),
            tested: AtomicU64::new(0),
            start: Instant::now(),
        }
    }

    pub fn stopped(&self) -> bool {
        self.state.load(Ordering::Acquire) != 0
    }

    pub fn cancel(&self) {
        let _ = self
            .state
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire);
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
    /// even if subsequent file output fails; another winner must not be written.
    pub fn claim_verified_winner(&self) -> bool {
        self.state
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
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
        // Saturate rather than wrapping statistics on an extremely long search.
        let _ = self
            .tested
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                Some(old.saturating_add(count))
            });
    }

    pub fn statistics(&self) -> (u64, Duration) {
        (self.tested.load(Ordering::Relaxed), self.start.elapsed())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

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
