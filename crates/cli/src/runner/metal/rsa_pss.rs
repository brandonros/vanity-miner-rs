//! RSA-PSS specialization of the dynamic-message candidate transport.
use super::candidate::{CandidateTransport, Contract};
use logic::{
    modes::rsa_pss::RsaPssRequest,
    search::{candidate_result::CandidateResult, hex_pattern::HexPattern},
};

pub struct RsaPss;
// SAFETY: the matching stock-Rust entry uses the six-buffer contract, guards
// padded lanes, bounds audit writes by count, and atomically claims the winner.
unsafe impl Contract for RsaPss {
    type Request = RsaPssRequest;
    const INTERFACE: &'static str =
        include_str!("../../../../kernels/rsa-pss/kernel.interface.json");
    fn candidate(
        request: &RsaPssRequest,
        pattern: &HexPattern,
        message: &[u8],
        counter: u64,
    ) -> CandidateResult {
        logic::modes::rsa_pss::rsa_pss(request, message, counter, pattern)
    }
}
pub type RsaPssTransport = CandidateTransport<RsaPss>;
