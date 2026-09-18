//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::modes::p256_signature::P256SignatureRequest;
pub type Pattern = logic::search::hex_pattern::HexPattern;
pub struct P256Signature;
// SAFETY: the explicit entry delegates all six-buffer mechanics to candidate_entry.
unsafe impl Contract for P256Signature {
    type Request = Request;
    type Pattern = Pattern;
    const ENTRY: &'static str = "kernel_p256_signature_vanity";
    fn candidate(
        request: &Request,
        pattern: &Pattern,
        payload: &[u8],
        counter: u64,
    ) -> CandidateResult {
        let _ = payload;
        logic::modes::p256_signature::p256_signature(request, payload, counter, pattern)
    }
}
