//! Shared mode-test helpers: CPU batch evaluation and exported-record assertions.
use logic::search::candidate_result::{BatchResult, CandidateResult};

#[cfg(any(
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-pss"
))]
pub(crate) fn assert_continuous(
    expected: usize,
    run: impl FnOnce(std::sync::Arc<crate::runner::session::SearchControl>) -> Result<(), String>,
) {
    use crate::runner::{progress::GlobalStats, session::SearchControl};
    use std::sync::Arc;
    let stats = Arc::new(GlobalStats::new(1, 0, 0));
    let control = Arc::new(SearchControl::with_stats(stats.clone()));
    control.set_continuous_device();
    control.set_batch_size(1).unwrap();
    control.set_device_launch_limit(Some(2));
    run(control.clone()).unwrap();
    assert_eq!(stats.matches(), expected);
    assert_eq!(stats.statistics().0, expected as u64);
    assert!(
        !control.has_winner(),
        "streaming output must not stop peer devices"
    );
}

pub(crate) fn evaluate_batch(
    start: u64,
    count: u32,
    mut candidate: impl FnMut(u64) -> CandidateResult,
) -> Result<BatchResult, String> {
    let mut output = BatchResult::EMPTY;
    for lane in 0..count {
        let counter = start
            .checked_add(lane as u64)
            .ok_or("test counter overflow")?;
        let result = candidate(counter);
        match result.status {
            CandidateResult::STATUS_MISS => {}
            CandidateResult::STATUS_MATCH => {
                if output.matches == 0 {
                    output.lane = lane;
                    output.candidate = result;
                }
                output.matches += 1;
            }
            _ => output.errors += 1,
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_counts_all_matches_but_keeps_only_first_payload() {
        let result = evaluate_batch(10, 4, |counter| {
            if counter == 10 {
                CandidateResult::MISS
            } else {
                CandidateResult::matched(&[counter as u8])
            }
        })
        .unwrap();
        assert_eq!(result.matches, 3);
        let (lane, candidate) = result.winner(4).unwrap().unwrap();
        assert_eq!(lane, 1);
        assert_eq!(candidate.bytes[0], 11);
    }

    #[test]
    fn batch_still_observes_errors_after_winning_lane() {
        let result = evaluate_batch(0, 2, |counter| {
            if counter == 0 {
                CandidateResult::matched(&[42])
            } else {
                CandidateResult::ERROR
            }
        })
        .unwrap();
        assert_eq!(result.matches, 1);
        assert_eq!(result.errors, 1);
        assert!(result.winner(2).is_err());
    }
}

#[cfg(test)]
pub fn console_field(record: &str, name: &str) -> Vec<u8> {
    let value = record
        .lines()
        .find_map(|line| {
            let (_, field) = line.split_once("] ")?;
            let (key, value) = field.split_once('=')?;
            (key == name).then_some(value)
        })
        .expect("missing console output field");
    hex::decode(value).unwrap()
}
