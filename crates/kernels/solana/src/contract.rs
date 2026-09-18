//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::search::xoroshiro::BatchSeed;
pub type Pattern = logic::search::vanity::BytePattern;
pub struct Solana;
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
    pub unsafe extern "C" fn kernel_solana_vanity(
        launch: Read Fixed Launch,
        request: Read Fixed Request,
        pattern: Read Fixed Pattern,
        message: Read Slice u8,
        output: ReadWrite Fixed BatchResult,
        records: Write Slice CandidateResult,
    ) {
        unsafe { entry::dispatch::<Solana>(launch, request, pattern, message, output, records); }
    } dispatch Grid1d;
}

// SAFETY: the explicit entry delegates the shared six-buffer mechanics to candidate_entry.
unsafe impl Contract for Solana {
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
