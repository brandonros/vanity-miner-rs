//! Bounded batching, cancellation, and verified winner selection.
use crate::search_control::SearchControl;
use logic::search::candidate_result::{BatchResult, CandidateResult};
use zeroize::Zeroizing;

/// Verify the GPU-selected winner before claiming the output. Rejected winners
/// discard the rest of that batch, matching the original shared-winner protocol.
/// Fixed batches make cancellation observable between launches.
pub fn find(
    mut evaluate: impl FnMut(u64, u32) -> Result<BatchResult, String>,
    limit: u64,
    control: &SearchControl,
    mut verify: impl FnMut(u64, &[u8; 256]) -> Result<bool, String>,
) -> Result<Option<(u64, CandidateResult)>, String> {
    let stop = control.cancel_on_exit();
    while let Some(batch) = control.reserve_bounded_batch(u64::from(control.batch_size()), limit) {
        let count = (batch.end - batch.start) as u32;
        let results = Zeroizing::new(evaluate(batch.start, count)?);
        let winner = results.winner(count)?;
        control.add_tested(count as u64);
        if control.stopped() {
            return Ok(None);
        }
        if let Some((lane, result)) = winner {
            let counter = batch.start + lane as u64;
            if verify(counter, &result.bytes)? && control.claim_verified_winner() {
                return Ok(Some((counter, result)));
            }
        }
    }
    stop.finish();
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_batches_cover_the_tail_without_overlap() {
        let control = SearchControl::new();
        control.set_batch_size(4096).unwrap();
        let mut batches = Vec::new();
        assert!(
            find(
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
        let found = find(
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
                Ok(counter == 64)
            },
        )
        .unwrap();
        assert_eq!(found.unwrap().0, 64);
        assert_eq!(batches, [(0, 64), (64, 1)]);
        assert_eq!(verified, [0, 64]);
        assert_eq!(control.statistics().0, 65);
        assert!(control.stopped());
    }

    #[test]
    fn malformed_batch_cancels_without_verifying() {
        let control = SearchControl::new();
        let result = find(
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
        let result = find(
            |_, _| panic!("cancelled search launched work"),
            1,
            &control,
            |_, _| panic!("cancelled search verified a result"),
        );
        assert!(result.unwrap().is_none());
    }
}
