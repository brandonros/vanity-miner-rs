//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::search::xoroshiro::BatchSeed;
pub type Pattern = [u8; 32];
pub struct Shallenge;
// SAFETY: the explicit entry delegates the shared six-buffer mechanics to candidate_entry.
unsafe impl Contract for Shallenge {
    type Request = Request;
    type Pattern = Pattern;
    const ENTRY: &'static str = "kernel_shallenge";
    fn candidate(
        request: &Request,
        pattern: &Pattern,
        payload: &[u8],
        counter: u64,
    ) -> CandidateResult {
        let _ = payload;
        logic::modes::shallenge::candidate(request, counter, pattern, payload)
    }
    fn validate_payload(payload: &[u8]) -> Result<(), &'static str> {
        if payload.len() > 32 {
            Err("invalid Metal username storage")
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn target_matching_and_invalid_requests() {
        let mut request = Request {
            seed: 12345,
            width: 32,
        };
        assert_eq!(
            Shallenge::candidate(&request, &[255; 32], b"aaaaaaaaaa", 0).status,
            CandidateResult::STATUS_MATCH
        );
        request.width = 0;
        assert_eq!(
            Shallenge::candidate(&request, &[255; 32], b"aaaaaaaaaa", 0).status,
            CandidateResult::STATUS_ERROR
        );
        request.width = 32;
        for length in [0, 31, 32] {
            assert_eq!(
                Shallenge::candidate(&request, &[255; 32], &[b'a'; 32][..length], 0).status,
                CandidateResult::STATUS_ERROR
            );
        }
        assert!(Shallenge::validate_payload(&[b'a'; 33]).is_err());
    }
}
