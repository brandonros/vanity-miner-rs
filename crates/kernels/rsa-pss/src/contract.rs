//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::modes::rsa_pss::RsaPssRequest;
pub type Pattern = logic::search::hex_pattern::HexPattern;
pub struct RsaPss;
// SAFETY: the explicit entry delegates the shared six-buffer mechanics to candidate_entry.
unsafe impl Contract for RsaPss {
    type Request = Request;
    type Pattern = Pattern;
    const ENTRY: &'static str = "kernel_rsa_pss_signature_vanity";
    fn candidate(
        request: &Request,
        pattern: &Pattern,
        payload: &[u8],
        counter: u64,
    ) -> CandidateResult {
        let _ = payload;
        logic::modes::rsa_pss::rsa_pss(request, payload, counter, pattern)
    }
}
