//! Independent RSA modulus attempts using the shared candidate transport.
use super::candidate::{CandidateTransport, Contract};
use logic::{
    modes::rsa_modulus::SearchConfig,
    search::{candidate_result::CandidateResult, hex_pattern::HexPattern},
};

pub struct RsaModulus;
// SAFETY: the matching stock-Rust entry uses the six-buffer contract, guards
// padded lanes, bounds audit writes by count, and atomically claims the winner.
unsafe impl Contract for RsaModulus {
    type Request = SearchConfig;
    const INTERFACE: &'static str =
        include_str!("../../../../kernels/rsa-modulus/metal/kernel.interface.json");
    fn candidate(
        request: &SearchConfig,
        pattern: &HexPattern,
        _message: &[u8],
        counter: u64,
    ) -> CandidateResult {
        logic::modes::rsa_modulus::rsa_modulus(request, counter, pattern)
    }
}
pub type RsaTransport = CandidateTransport<RsaModulus>;
