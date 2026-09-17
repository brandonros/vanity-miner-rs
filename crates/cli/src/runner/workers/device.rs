//! Each device owns its state and search until completion or cancellation.
use crate::runner::session::SearchControl;
use std::{error::Error, sync::Arc};

pub type RunResult = Result<(), Box<dyn Error + Send + Sync>>;

pub fn run<State>(
    count: usize,
    control: Arc<SearchControl>,
    initialize: impl Fn(usize) -> Result<State, Box<dyn Error + Send + Sync>> + Sync,
    evaluate: impl Fn(&mut State, usize) -> RunResult + Sync,
) -> RunResult {
    if count == 0 {
        return Err("no device workers requested".into());
    }
    let control = &*control;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for ordinal in 0..count {
            let (initialize, evaluate) = (&initialize, &evaluate);
            handles.push(scope.spawn(move || {
                let stop = control.cancel_on_exit();
                let mut state = initialize(ordinal)?;
                evaluate(&mut state, ordinal)?;
                stop.finish();
                Ok(())
            }));
        }
        let mut first = None;
        for handle in handles {
            let result: RunResult = handle
                .join()
                .unwrap_or_else(|_| Err("device worker panicked".into()));
            if let Err(error) = result {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::progress::GlobalStats;
    use std::{
        cell::Cell,
        rc::Rc,
        sync::atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn device_workers_finish_independently_without_winner_rounds() {
        let stats = Arc::new(GlobalStats::new(16, 0, 0));
        let control = Arc::new(SearchControl::with_stats(stats.clone()));
        control.set_continuous();
        let initialized = AtomicUsize::new(0);
        let finished = AtomicUsize::new(0);
        run(
            16,
            control.clone(),
            |_| {
                initialized.fetch_add(1, Ordering::SeqCst);
                Ok(Rc::new(Cell::new(0)))
            },
            |state, ordinal| {
                for _ in 0..=ordinal {
                    assert!(!control.stopped());
                    state.set(state.get() + 1);
                    control.add_verified_match();
                }
                assert_eq!(state.get(), ordinal + 1);
                finished.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(initialized.load(Ordering::SeqCst), 16);
        assert_eq!(finished.load(Ordering::SeqCst), 16);
        assert!(!control.has_winner());
        assert!(!control.stopped());
    }

    #[test]
    fn initialization_failure_and_worker_panic_cancel_peers() {
        for panic_during_work in [false, true] {
            let control = Arc::new(SearchControl::new());
            let result = run(
                3,
                control.clone(),
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
