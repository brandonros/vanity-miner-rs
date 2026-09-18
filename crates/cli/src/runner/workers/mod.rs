//! Own, cancel and join every CPU worker before delivering a verified result.
use crate::runner::session::SearchControl;

pub(crate) fn search<T: Send>(
    count: usize,
    control: &SearchControl,
    work: impl Fn(usize) -> Result<Option<T>, String> + Sync,
) -> Result<Option<T>, String> {
    if count == 0 {
        return Err("worker count must be nonzero".into());
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(count);
        let mut error = None;
        for id in 0..count {
            let work = &work;
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                let guard = control.cancel_on_exit();
                let result = work(id);
                if result.is_ok() {
                    guard.finish();
                }
                result
            }) {
                Ok(handle) => handles.push((id, handle)),
                Err(e) => {
                    control.cancel();
                    error = Some(format!("Could not spawn worker thread {id}: {e}"));
                    break;
                }
            }
        }
        let mut winner = None;
        for (id, handle) in handles {
            match handle.join() {
                Ok(Ok(Some(record))) => {
                    if winner.is_none() {
                        winner = Some(record);
                    }
                }
                Ok(Ok(None)) => (),
                Ok(Err(e)) => {
                    control.cancel();
                    error.get_or_insert(format!("Worker thread {id} failed: {e}"));
                }
                Err(_) => {
                    control.cancel();
                    error.get_or_insert(format!("Worker thread {id} panicked"));
                }
            }
        }
        error.map_or(Ok(winner), Err)
    })
}

pub mod device;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::{Duration, Instant};

    fn later_failure(panic: bool) {
        let ready = Barrier::new(3);
        let finished = AtomicUsize::new(0);
        let control = SearchControl::new();
        let result = search::<()>(3, &control, |id| {
            ready.wait();
            if id == 1 {
                if panic {
                    panic!("injected failure");
                }
                return Err("injected failure".into());
            }
            let deadline = Instant::now() + Duration::from_secs(5);
            while !control.stopped() {
                assert!(Instant::now() < deadline, "peer cancellation timed out");
                std::thread::yield_now();
            }
            finished.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        });
        assert_eq!(
            result.unwrap_err(),
            if panic {
                "Worker thread 1 panicked"
            } else {
                "Worker thread 1 failed: injected failure"
            }
        );
        assert_eq!(finished.load(Ordering::Relaxed), 2);
    }
    #[test]
    fn later_error_cancels_and_joins_all_peers() {
        later_failure(false);
    }
    #[test]
    fn later_panic_cancels_and_joins_all_peers() {
        later_failure(true);
    }
    #[test]
    fn winner_is_delivered_after_all_workers_join() {
        let control = SearchControl::new();
        control.set_exit_on_first_match();
        let ready = Barrier::new(8);
        let finished = AtomicUsize::new(0);
        let result = search(8, &control, |id| {
            ready.wait();
            let winner = control.claim_verified_winner().then_some(id);
            finished.fetch_add(1, Ordering::Relaxed);
            Ok(winner)
        })
        .unwrap();
        assert!(result.is_some());
        assert_eq!(finished.load(Ordering::Relaxed), 8);
        assert!(!control.resume_after_match());
    }
    #[test]
    fn finite_worker_does_not_cancel_pending_partitions() {
        let control = SearchControl::new();
        let ready = Barrier::new(4);
        let finished = AtomicUsize::new(0);
        let result = search::<()>(4, &control, |_| {
            ready.wait();
            while let Some(batch) = control.reserve_bounded_batch(1, 257) {
                assert_eq!(batch.end - batch.start, 1);
                finished.fetch_add(1, Ordering::Relaxed);
            }
            Ok(None)
        })
        .unwrap();
        assert!(result.is_none());
        assert_eq!(finished.load(Ordering::Relaxed), 257);
        assert!(!control.stopped());
        assert!(search::<()>(0, &control, |_| Ok(None)).is_err());
    }
}
