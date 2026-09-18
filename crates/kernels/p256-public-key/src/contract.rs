//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::modes::p256_public_key::P256PublicRequest;
pub type Pattern = logic::search::hex_pattern::HexPattern;
pub struct P256Public;
// SAFETY: the explicit entry delegates the shared six-buffer mechanics to candidate_entry.
unsafe impl Contract for P256Public {
    type Request = Request;
    type Pattern = Pattern;
    const ENTRY: &'static str = "kernel_p256_public_key_vanity";
    fn candidate(
        request: &Request,
        pattern: &Pattern,
        payload: &[u8],
        counter: u64,
    ) -> CandidateResult {
        let _ = payload;
        logic::modes::p256_public_key::p256_public(request, counter, pattern)
    }
}
