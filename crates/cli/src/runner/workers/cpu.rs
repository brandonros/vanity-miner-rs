use crate::runner::session::SearchControl;
use std::error::Error;
use std::sync::Arc;

type WorkerResult = Result<(), Box<dyn Error + Send + Sync>>;

/// Workers must check cancellation between units of work. Every started worker
/// is joined before returning, including after a worker or thread-spawn failure.
pub fn spawn_cpu_workers<T, F>(
    num_workers: usize,
    shared_data: Arc<T>,
    worker_fn: F,
) -> WorkerResult
where
    T: Send + Sync + 'static,
    F: Fn(usize, Arc<T>, Arc<SearchControl>) -> WorkerResult + Send + Clone + 'static,
{
    let cancelled = Arc::new(SearchControl::new());
    let mut handles = Vec::with_capacity(num_workers);
    let mut first_error: Option<Box<dyn Error + Send + Sync>> = None;
    for i in 0..num_workers {
        let data = Arc::clone(&shared_data);
        let f = worker_fn.clone();
        let stop = Arc::clone(&cancelled);
        match std::thread::Builder::new().spawn(move || {
            let _guard = stop.cancel_on_exit();
            f(i, data, Arc::clone(&stop))
        }) {
            Ok(handle) => handles.push(handle),
            Err(error) => {
                cancelled.cancel();
                first_error = Some(format!("Could not spawn worker thread {i}: {error}").into());
                break;
            }
        }
    }

    for (i, handle) in handles.into_iter().enumerate() {
        let error = match handle.join() {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(format!("Worker thread {i} failed: {error}")),
            Err(_) => Some(format!("Worker thread {i} panicked")),
        };
        if let Some(error) = error {
            first_error.get_or_insert_with(|| error.into());
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::{Duration, Instant};

    fn later_worker_failure(panic: bool) {
        let ready = Arc::new(Barrier::new(3));
        let peers_finished = Arc::new(AtomicUsize::new(0));
        let finished = peers_finished.clone();
        let result = spawn_cpu_workers(3, ready, move |id, ready, stop| {
            ready.wait();
            if id == 1 {
                if panic {
                    panic!("injected failure");
                }
                return Err("injected failure".into());
            }
            // Bound failure time so a broken cancellation signal fails the test
            // instead of leaving an endless worker or hanging the test suite.
            let deadline = Instant::now() + Duration::from_secs(5);
            while !stop.stopped() {
                assert!(Instant::now() < deadline, "peer cancellation timed out");
                std::thread::yield_now();
            }
            finished.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });
        let error = result.unwrap_err().to_string();
        assert_eq!(
            error,
            if panic {
                "Worker thread 1 panicked"
            } else {
                "Worker thread 1 failed: injected failure"
            }
        );
        assert_eq!(peers_finished.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn later_error_cancels_and_joins_all_peers() {
        later_worker_failure(false);
    }

    #[test]
    fn later_panic_cancels_and_joins_all_peers() {
        later_worker_failure(true);
    }
}
