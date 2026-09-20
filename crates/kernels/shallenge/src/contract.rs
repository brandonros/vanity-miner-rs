//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::search::xoroshiro::BatchSeed;
pub type Pattern = [u8; 32];
pub struct Shallenge;
#[macro_use]
#[path = "../../common/candidate_bindings.rs"]
mod bindings;
#[cfg(target_arch = "nvptx64")]
#[path = "../../common/candidate_entry.rs"]
mod entry;
use logic::search::{candidate_abi::Launch, candidate_result::BatchResult};
logic::llvm_metal_kernel::kernel! {
    pub mod abi;
    /// # Safety
    /// Disjoint buffers, valid dynamic spans and synchronized result ownership.
    #[cfg(target_arch = "nvptx64")]
    pub unsafe extern "C" fn kernel_shallenge(
        launch: Read Fixed Launch,
        request: Read Fixed Request,
        pattern: Read Fixed Pattern,
        message: Read Slice u8,
        output: ReadWrite Fixed BatchResult,
        records: Write Slice CandidateResult,
    ) {
        unsafe { entry::dispatch::<Shallenge>(launch, request, pattern, message, output, records); }
    } dispatch Grid1d;
}

// SAFETY: the explicit entry delegates the shared six-buffer mechanics to candidate_entry.
unsafe impl Contract for Shallenge {
    type Request = Request;
    type Pattern = Pattern;
    candidate_bindings!();
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
