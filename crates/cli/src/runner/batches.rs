//! Bounded batching, cancellation, and verified result delivery.
use crate::runner::session::SearchControl;
use logic::search::candidate_result::BatchResult;
use zeroize::Zeroizing;

/// One evaluation and verification path for both first-match and continuous
/// searches. Continuous searches overlap device work with host verification.
pub fn search(
    mut evaluate: impl FnMut(u64, u32) -> Result<BatchResult, String>,
    limit: u64,
    control: &SearchControl,
    mut format_verified: impl FnMut(u64, &[u8; 256]) -> Result<Option<String>, String> + Send,
) -> Result<Option<String>, String> {
    let stop = control.cancel_on_exit();
    let mut output = None;
    let mut produce = || {
        let Some(batch) = control.reserve_bounded_batch(u64::from(control.batch_size()), limit)
        else {
            return Ok(None);
        };
        if !control.reserve_device_launch() {
            return Ok(None);
        }
        let count = (batch.end - batch.start) as u32;
        let result = Zeroizing::new(evaluate(batch.start, count)?);
        result.winner(count)?;
        control.add_tested(u64::from(count));
        Ok(Some((batch.start, count, result)))
    };
    let mut consume = |(start, count, result): (u64, u32, Zeroizing<BatchResult>)| {
        if let Some((lane, candidate)) = result.winner(count)? {
            let candidate = Zeroizing::new(candidate);
            if let Some(record) = format_verified(start + u64::from(lane), &candidate.bytes)? {
                if control.continuous() {
                    crate::runner::progress::print_verified(control, record)?;
                } else if control.claim_verified_winner() {
                    output = Some(record);
                }
            }
        }
        Ok(())
    };
    if control.continuous() {
        pump(control, produce, consume)?;
    } else {
        // First-match callers must observe verification before reserving the
        // next batch, so launch limits cannot discard the last verified winner.
        while !control.stopped() {
            let Some(batch) = produce()? else {
                break;
            };
            consume(batch)?;
        }
    }
    stop.finish();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use logic::search::candidate_result::CandidateResult;

    #[test]
    fn launch_limit_keeps_last_winner_and_persists_across_rounds() {
        let control = SearchControl::new();
        control.set_device_launch_limit(Some(1));
        let found = search(
            |_, _| {
                Ok(BatchResult {
                    matches: 1,
                    errors: 0,
                    lane: 0,
                    candidate: CandidateResult::matched(&[42]),
                })
            },
            1024,
            &control,
            |_, _| Ok(Some("winner".into())),
        )
        .unwrap();
        assert!(found.is_some());
        assert!(control.resume_after_match());
        assert!(
            search(
                |_, _| panic!("launch limit must persist after a match"),
                1024,
                &control,
                |_, _| panic!("no further candidates"),
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(control.statistics().0, 64);
    }

    #[test]
    fn launch_limit_without_match_is_normal_completion() {
        let control = SearchControl::new();
        control.set_device_launch_limit(Some(2));
        let mut launches = 0;
        assert!(
            search(
                |_, _| {
                    launches += 1;
                    Ok(BatchResult::EMPTY)
                },
                1024,
                &control,
                |_, _| panic!("no match"),
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(launches, 2);
        assert_eq!(control.statistics().0, 128);
    }

    #[test]
    fn large_batches_cover_the_tail_without_overlap() {
        let control = SearchControl::new();
        control.set_batch_size(4096).unwrap();
        let mut batches = Vec::new();
        assert!(
            search(
                |start, count| {
                    batches.push((start, count));
                    Ok(BatchResult::EMPTY)
                },
                8200,
                &control,
                |_, _| panic!("no winner")
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(batches, [(0, 4096), (4096, 4096), (8192, 8)]);
        assert_eq!(control.statistics().0, 8200);
        assert!(control.set_batch_size(0).is_err());
        assert!(control.set_batch_size(1_048_577).is_err());
    }

    #[test]
    fn rejected_winner_advances_to_next_batch_without_replay() {
        let control = SearchControl::new();
        let mut batches = Vec::new();
        let mut verified = Vec::new();
        let found = search(
            |start, count| {
                batches.push((start, count));
                Ok(BatchResult {
                    matches: count,
                    errors: 0,
                    lane: 0,
                    candidate: CandidateResult::matched(&[1]),
                })
            },
            65,
            &control,
            |counter, _| {
                verified.push(counter);
                Ok((counter == 64).then(|| counter.to_string()))
            },
        )
        .unwrap();
        assert_eq!(found.unwrap(), "64");
        assert_eq!(batches, [(0, 64), (64, 1)]);
        assert_eq!(verified, [0, 64]);
        assert_eq!(control.statistics().0, 65);
        assert!(control.stopped());
    }

    #[test]
    fn malformed_batch_cancels_without_verifying() {
        let control = SearchControl::new();
        let result = search(
            |_, _| {
                Ok(BatchResult {
                    matches: 2,
                    ..BatchResult::EMPTY
                })
            },
            1,
            &control,
            |_, _| panic!("must not verify a malformed batch"),
        );
        assert!(matches!(result, Err(error) if error.contains("batch counts")));
        assert!(control.stopped());
        assert_eq!(control.statistics().0, 0);
    }

    #[test]
    fn cancelled_search_never_evaluates_a_batch() {
        let control = SearchControl::new();
        control.cancel();
        let result = search(
            |_, _| panic!("cancelled search launched work"),
            1,
            &control,
            |_, _| panic!("cancelled search verified a result"),
        );
        assert!(result.unwrap().is_none());
    }
}

pub fn pump<T: Send>(
    control: &SearchControl,
    mut produce: impl FnMut() -> Result<Option<T>, String>,
    mut consume: impl FnMut(T) -> Result<(), String> + Send,
) -> Result<(), String> {
    let stop = control.cancel_on_exit();
    let result = std::thread::scope(|scope| {
        // Two pending batches plus the batch being verified. Backpressure keeps
        // memory bounded even when every candidate matches an easy pattern.
        let (sender, receiver) = std::sync::mpsc::sync_channel(2);
        let verifier = scope.spawn(move || {
            let stop = control.cancel_on_exit();
            for batch in receiver {
                consume(batch)?;
            }
            stop.finish();
            Ok::<_, String>(())
        });
        let produced = (|| {
            while !control.stopped() {
                let Some(batch) = produce()? else {
                    break;
                };
                if sender.send(batch).is_err() {
                    break;
                }
            }
            Ok::<_, String>(())
        })();
        if produced.is_err() {
            control.cancel();
        }
        drop(sender);
        let verified = verifier
            .join()
            .unwrap_or_else(|_| Err("device result verifier panicked".into()));
        verified.and(produced)
    });
    if result.is_ok() {
        stop.finish();
    }
    result
}

#[cfg(test)]
mod pipeline_tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    #[test]
    fn producer_advances_while_first_result_is_being_verified() {
        let control = SearchControl::new();
        let (advanced, observe) = mpsc::channel();
        let mut produced = 0;
        let mut verified = Vec::new();
        let verified_ref = &mut verified;
        pump(
            &control,
            || {
                if produced == 5 {
                    return Ok(None);
                }
                let value = produced;
                produced += 1;
                if value == 1 {
                    advanced.send(()).unwrap();
                }
                Ok(Some(value))
            },
            move |value| {
                if value == 0 {
                    observe
                        .recv_timeout(Duration::from_secs(2))
                        .expect("producer stalled on verification");
                }
                verified_ref.push(value);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(verified, [0, 1, 2, 3, 4]);
        assert!(!control.stopped());
    }

    #[test]
    fn failures_cancel_and_join_with_a_full_queue() {
        for panic in [false, true] {
            let control = SearchControl::new();
            let result = pump(
                &control,
                || Ok(Some(42)),
                |_| {
                    assert!(!panic, "injected verifier panic");
                    Err("injected verification failure".into())
                },
            );
            assert!(result.is_err());
            assert!(control.stopped());
        }
        let control = SearchControl::new();
        let result = pump::<u32>(
            &control,
            || Err("producer failed".into()),
            |_| panic!("no result"),
        );
        assert!(result.is_err());
        assert!(control.stopped());
    }

    #[test]
    fn cancellation_prevents_launching_and_drains_queued_results() {
        let control = SearchControl::new();
        control.cancel();
        pump::<u32>(
            &control,
            || panic!("cancelled producer ran"),
            |_| panic!("no results"),
        )
        .unwrap();
        let control = SearchControl::new();
        let mut count = 0;
        pump(
            &control,
            || {
                control.cancel();
                Ok(Some(42))
            },
            |value| {
                assert_eq!(value, 42);
                count += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(count, 1);
    }
}
