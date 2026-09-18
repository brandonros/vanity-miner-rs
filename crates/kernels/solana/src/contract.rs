//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::search::xoroshiro::BatchSeed;
pub type Pattern = logic::search::vanity::BytePattern;
pub struct Solana;
// SAFETY: the explicit entry delegates the shared six-buffer mechanics to candidate_entry.
unsafe impl Contract for Solana {
    type Request = Request;
    type Pattern = Pattern;
    const ENTRY: &'static str = "kernel_solana_vanity";
    fn candidate(
        request: &Request,
        pattern: &Pattern,
        payload: &[u8],
        counter: u64,
    ) -> CandidateResult {
        let _ = payload;
        logic::modes::solana::candidate(request, counter, pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matching_and_invalid_requests() {
        let mut seed = Request {
            seed: 583437459223573146,
            width: 32,
        };
        let mut pattern = Pattern::new(b"aaa", b"NFC").unwrap();
        let result = Solana::candidate(&seed, &pattern, &[], 3);
        assert_eq!(result.status, CandidateResult::STATUS_MATCH);
        assert_eq!(
            &result.bytes[..32],
            &[
                0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48, 0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3,
                0xd2, 0xc4, 0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14, 0xb0, 0xff, 0x7c, 0x5a,
                0xb5, 0xe7, 0x95, 0x6b,
            ]
        );
        pattern.suffix[0] ^= 1;
        assert_eq!(
            Solana::candidate(&seed, &pattern, &[], 3).status,
            CandidateResult::STATUS_MISS
        );
        pattern.prefix_len = 65;
        assert_eq!(
            Solana::candidate(&seed, &pattern, &[], 3).status,
            CandidateResult::STATUS_ERROR
        );
        pattern = Pattern::new(&[], &[]).unwrap();
        seed.width = 0;
        assert_eq!(
            Solana::candidate(&seed, &pattern, &[], 3).status,
            CandidateResult::STATUS_ERROR
        );
    }
}
