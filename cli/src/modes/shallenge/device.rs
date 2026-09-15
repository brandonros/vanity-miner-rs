//! Shared nonce verification and best-hash publication for device backends.
use super::shared_best_hash::SharedBestHash;
use crate::runner::{batches, session::SearchControl};
use logic::search::{candidate_result::BatchResult, xoroshiro::BatchSeed};
use std::sync::{Arc, RwLock};

pub const ENTRY: &str = "kernel_shallenge";

pub fn search(
    username: &str,
    best: Arc<RwLock<SharedBestHash>>,
    seed: Option<u64>,
    control: &SearchControl,
    mut evaluate: impl FnMut(&BatchSeed, &[u8; 32], &[u8], u64, u32) -> Result<BatchResult, String>,
) -> Result<Option<String>, String> {
    let request = BatchSeed {
        seed: seed.unwrap_or_else(rand::random),
        width: u64::from(control.batch_size()),
    };
    let initial = best.read().unwrap_or_else(|e| e.into_inner()).get_current();
    batches::search(
        |start, count| {
            let target = best.read().unwrap_or_else(|e| e.into_inner()).get_current();
            evaluate(&request, &target, username.as_bytes(), start, count)
        },
        u64::MAX,
        control,
        |counter, bytes| {
            let expected = zeroize::Zeroizing::new(logic::modes::shallenge::candidate(
                &request,
                counter,
                &initial,
                username.as_bytes(),
            ));
            if expected.status != 1 || expected.bytes != *bytes {
                return Err("device nonce failed CPU verification".into());
            }
            let hash: [u8; 32] = bytes[..32].try_into().unwrap();
            let length = u32::from_le_bytes(bytes[96..100].try_into().unwrap()) as usize;
            let nonce = bytes[32..96].get(..length).ok_or("invalid nonce length")?;
            let nonce = std::str::from_utf8(nonce).map_err(|e| e.to_string())?;
            if !best
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .update_if_better(hash)
            {
                return Ok(None);
            }
            Ok(Some(format!(
                "[shallenge] hash={}\n[shallenge] nonce={}\n[shallenge] challenge={}/{}",
                hex::encode(hash),
                nonce,
                username,
                nonce
            )))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use logic::search::candidate_result::CandidateResult;

    #[test]
    fn malformed_nonce_never_changes_the_shared_target() {
        let best = Arc::new(RwLock::new(SharedBestHash::new([0xff; 32])));
        let control = SearchControl::new();
        control.set_batch_size(1).unwrap();
        let error = search(
            "miner",
            best.clone(),
            Some(1),
            &control,
            |r, p, m, start, _| {
                let mut candidate = logic::modes::shallenge::candidate(r, start, p, m);
                assert_eq!(candidate.status, CandidateResult::STATUS_MATCH);
                candidate.bytes[96..100].copy_from_slice(&u32::MAX.to_le_bytes());
                Ok(BatchResult {
                    matches: 1,
                    errors: 0,
                    lane: 0,
                    candidate,
                })
            },
        )
        .unwrap_err();
        assert!(error.contains("verification"));
        assert_eq!(best.read().unwrap().get_current(), [0xff; 32]);
    }

    #[test]
    fn verified_nonce_updates_target_and_exports_one_record() {
        let best = Arc::new(RwLock::new(SharedBestHash::new([0xff; 32])));
        let control = SearchControl::new();
        control.set_batch_size(1).unwrap();
        let record = search(
            "miner",
            best.clone(),
            Some(1),
            &control,
            |r, p, m, start, _| {
                Ok(BatchResult {
                    matches: 1,
                    errors: 0,
                    lane: 0,
                    candidate: logic::modes::shallenge::candidate(r, start, p, m),
                })
            },
        )
        .unwrap()
        .unwrap();
        assert!(record.contains(&hex::encode(best.read().unwrap().get_current())));
        assert!(best.read().unwrap().get_current() < [0xff; 32]);
        assert!(control.has_winner());
    }
}
