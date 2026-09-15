//! Persistent thread-owned device state with synchronized winner rounds.
use crate::{search_control::SearchControl, stats::GlobalStats};
use std::{
    error::Error,
    sync::{Arc, mpsc},
};

pub type RunResult = Result<(), Box<dyn Error + Send + Sync>>;

/// State is created and destroyed on its worker thread; it need not be Send.
/// No worker starts a new round until every peer has completed the previous one.
pub fn run<State>(
    count: usize,
    control: Arc<SearchControl>,
    stats: Arc<GlobalStats>,
    continuous: bool,
    initialize: impl Fn(usize) -> Result<State, Box<dyn Error + Send + Sync>> + Sync,
    evaluate: impl Fn(&mut State, usize) -> RunResult + Sync,
) -> RunResult {
    if count == 0 {
        return Err("no device workers requested".into());
    }
    std::thread::scope(|scope| {
        let mut workers = Vec::new();
        for ordinal in 0..count {
            let control = control.clone();
            let initialize = &initialize;
            let evaluate = &evaluate;
            let (request, requests) = mpsc::channel::<()>();
            let (response, responses) = mpsc::channel::<RunResult>();
            let handle = scope.spawn(move || -> RunResult {
                let stop = control.cancel_on_exit();
                let mut state = initialize(ordinal)?;
                while requests.recv().is_ok() {
                    let result = evaluate(&mut state, ordinal);
                    if result.is_err() {
                        control.cancel();
                    }
                    let failed = result.is_err();
                    if response.send(result).is_err() || failed {
                        break;
                    }
                }
                stop.finish();
                Ok(())
            });
            workers.push((request, responses, handle));
        }
        let mut first_error: Option<Box<dyn Error + Send + Sync>> = None;
        loop {
            for (request, _, _) in &workers {
                if request.send(()).is_err() {
                    control.cancel();
                }
            }
            for (_, responses, _) in &workers {
                match responses.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        control.cancel();
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                    Err(_) => {
                        control.cancel();
                        if first_error.is_none() {
                            first_error = Some("device worker stopped unexpectedly".into());
                        }
                    }
                }
            }
            if first_error.is_some() || !continuous || !control.has_winner() {
                break;
            }
            stats.add_matches(1);
            if !control.resume_after_match() {
                break;
            }
        }
        // Disconnect all workers before joining any of them.
        let handles: Vec<_> = workers
            .into_iter()
            .map(|(request, _, handle)| {
                drop(request);
                handle
            })
            .collect();
        for handle in handles {
            let result = handle
                .join()
                .unwrap_or_else(|_| Err("device worker panicked".into()));
            if let Err(error) = result {
                control.cancel();
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::Cell,
        rc::Rc,
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn non_send_state_is_reused_across_winner_rounds() {
        let stats = Arc::new(GlobalStats::new(3, 0, 0));
        let control = Arc::new(SearchControl::with_stats(stats.clone()));
        let initialized = AtomicUsize::new(0);
        let evaluations = AtomicUsize::new(0);
        run(
            3,
            control.clone(),
            stats,
            true,
            |_| {
                initialized.fetch_add(1, Ordering::SeqCst);
                Ok(Rc::new(Cell::new(0)))
            },
            |state, _| {
                state.set(state.get() + 1);
                evaluations.fetch_add(1, Ordering::SeqCst);
                if state.get() == 4 {
                    control.interrupt();
                } else {
                    control.claim_verified_winner();
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(initialized.load(Ordering::SeqCst), 3);
        assert_eq!(evaluations.load(Ordering::SeqCst), 12);
    }

    #[test]
    fn initialization_failure_and_worker_panic_cancel_peers() {
        for panic_during_work in [false, true] {
            let control = Arc::new(SearchControl::new());
            let result = run(
                3,
                control.clone(),
                Arc::new(GlobalStats::new(3, 0, 0)),
                true,
                |ordinal| {
                    if ordinal == 1 && !panic_during_work {
                        return Err("init failure".into());
                    }
                    Ok(())
                },
                |_, ordinal| {
                    if ordinal == 1 {
                        panic!("worker failure");
                    }
                    while !control.stopped() {
                        std::thread::yield_now();
                    }
                    Ok(())
                },
            );
            assert!(result.is_err());
            assert!(control.stopped());
        }
    }

    #[test]
    fn evaluation_error_and_finite_completion_do_not_restart() {
        for fail in [false, true] {
            let control = Arc::new(SearchControl::new());
            let evaluations = AtomicUsize::new(0);
            let result = run(
                2,
                control,
                Arc::new(GlobalStats::new(2, 0, 0)),
                true,
                |_| Ok(()),
                |_, ordinal| {
                    evaluations.fetch_add(1, Ordering::SeqCst);
                    if fail && ordinal == 1 {
                        Err("evaluation failure".into())
                    } else {
                        Ok(())
                    }
                },
            );
            assert_eq!(result.is_err(), fail);
            assert_eq!(evaluations.load(Ordering::SeqCst), 2);
        }
    }
}
