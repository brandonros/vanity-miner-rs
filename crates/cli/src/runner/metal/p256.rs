//! P-256 contracts over the common structured-candidate transport.
use super::candidate::{CandidateTransport, Contract};
use logic::search::{candidate_result::CandidateResult, hex_pattern::HexPattern};

#[cfg(feature = "p256-public-key")]
pub struct PublicKey;
#[cfg(feature = "p256-public-key")]
// SAFETY: the matching stock-Rust entry uses the six-buffer contract, guards
// padded lanes, bounds audit writes by count, and atomically claims the winner.
unsafe impl Contract for PublicKey {
    type Request = logic::modes::p256_public_key::P256PublicRequest;
    const INTERFACE: &'static str =
        include_str!("../../../../kernels/p256-public-key/kernel.interface.json");
    fn candidate(
        request: &Self::Request,
        pattern: &HexPattern,
        _message: &[u8],
        counter: u64,
    ) -> CandidateResult {
        logic::modes::p256_public_key::p256_public(request, counter, pattern)
    }
}
#[cfg(feature = "p256-public-key")]
pub type P256PublicTransport = CandidateTransport<PublicKey>;

#[cfg(feature = "p256-signature")]
pub use crate::modes::p256_signature::metal::P256SignatureTransport;
