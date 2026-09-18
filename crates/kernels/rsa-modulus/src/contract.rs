//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::modes::rsa_modulus::SearchConfig;
pub type Pattern = logic::search::hex_pattern::HexPattern;
pub struct RsaModulus;
// SAFETY: the explicit entry delegates the shared six-buffer mechanics to candidate_entry.
unsafe impl Contract for RsaModulus {
    type Request = Request;
    type Pattern = Pattern;
    const ENTRY: &'static str = "kernel_rsa_modulus_candidate";
    fn candidate(
        request: &Request,
        pattern: &Pattern,
        payload: &[u8],
        counter: u64,
    ) -> CandidateResult {
        let _ = payload;
        logic::modes::rsa_modulus::rsa_modulus(request, counter, pattern)
    }
}
