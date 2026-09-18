//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::search::xoroshiro::BatchSeed;
pub type Pattern = logic::search::vanity::BytePattern;
pub struct Ethereum;
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
    pub unsafe extern "C" fn kernel_ethereum_vanity(
        launch: Read Fixed Launch,
        request: Read Fixed Request,
        pattern: Read Fixed Pattern,
        message: Read Slice u8,
        output: ReadWrite Fixed BatchResult,
        records: Write Slice CandidateResult,
    ) {
        unsafe { entry::dispatch::<Ethereum>(launch, request, pattern, message, output, records); }
    } dispatch Grid1d;
}

// SAFETY: the explicit entry delegates the shared six-buffer mechanics to candidate_entry.
unsafe impl Contract for Ethereum {
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
        logic::modes::ethereum::candidate(request, counter, pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matching_and_invalid_requests() {
        let mut seed = Request {
            seed: 10088153575472065218,
            width: 32,
        };
        let mut pattern = Pattern::new(&[0x55], &[0x02]).unwrap();
        let result = Ethereum::candidate(&seed, &pattern, &[], 0);
        assert_eq!(result.status, CandidateResult::STATUS_MATCH);
        assert_eq!(
            &result.bytes[..32],
            &[
                0x23, 0xa3, 0x3f, 0x35, 0x73, 0x7a, 0xb1, 0xab, 0xc1, 0x6c, 0xc1, 0xd1, 0x75, 0x55,
                0xc8, 0xdc, 0x75, 0x18, 0x33, 0xac, 0x76, 0xcf, 0x4b, 0xc9, 0xe3, 0x2f, 0xaf, 0x3d,
                0x73, 0x52, 0xe9, 0x30,
            ]
        );
        pattern.suffix[0] ^= 1;
        assert_eq!(
            Ethereum::candidate(&seed, &pattern, &[], 0).status,
            CandidateResult::STATUS_MISS
        );
        pattern.prefix_len = 65;
        assert_eq!(
            Ethereum::candidate(&seed, &pattern, &[], 0).status,
            CandidateResult::STATUS_ERROR
        );
        pattern = Pattern::new(&[], &[]).unwrap();
        seed.width = 0;
        assert_eq!(
            Ethereum::candidate(&seed, &pattern, &[], 0).status,
            CandidateResult::STATUS_ERROR
        );
    }
}
