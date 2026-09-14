//! Bounded batching, cancellation, and verified winner selection.
use crate::search_control::SearchControl;
use logic::candidate_result::CandidateResult;
use zeroize::Zeroizing;

/// Every candidate is reconstructed and verified before reserving the one output
/// slot. Fixed batches make cancellation observable between launches.
pub fn find(
    mut evaluate: impl FnMut(u64, u32) -> Result<Vec<CandidateResult>, String>,
    limit: u64,
    control: &SearchControl,
    mut verify: impl FnMut(u64, &[u8; 256]) -> Result<bool, String>,
) -> Result<Option<(u64, CandidateResult)>, String> {
    let stop = control.cancel_on_exit();
    while let Some(batch) = control.reserve_bounded_batch(64, limit) {
        let count = (batch.end - batch.start) as u32;
        let results = Zeroizing::new(evaluate(batch.start, count)?);
        if results.len() != count as usize {
            return Err("device returned an incorrect lane count".into());
        }
        control.add_tested(count as u64);
        for (lane, result) in results.iter().enumerate() {
            if control.stopped() {
                return Ok(None);
            }
            match result.status {
                0 => {}
                1 => {
                    let counter = batch.start + lane as u64;
                    if verify(counter, &result.bytes)? && control.claim_verified_winner() {
                        return Ok(Some((counter, *result)));
                    }
                }
                _ => return Err("device candidate evaluation failed".into()),
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
    fn finite_tail_and_rejected_matches_keep_candidate_order() {
        let control = SearchControl::new();
        let mut batches = Vec::new();
        let mut verified = Vec::new();
        let found = find(
            |start, count| {
                batches.push((start, count));
                Ok(vec![CandidateResult::matched(&[1]); count as usize])
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
        assert_eq!(verified, (0..65).collect::<Vec<_>>());
        assert_eq!(control.statistics().0, 65);
        assert!(control.stopped());
    }

    #[test]
    fn malformed_batch_cancels_without_verifying() {
        let control = SearchControl::new();
        let result = find(
            |_, _| Ok(Vec::new()),
            1,
            &control,
            |_, _| panic!("must not verify a malformed batch"),
        );
        assert!(matches!(result, Err(error) if error.contains("lane count")));
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
