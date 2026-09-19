//! Shared typed host/device contract.
use logic::search::{candidate_abi::Contract, candidate_result::CandidateResult};
pub type Request = logic::modes::rsa_pss::RsaPssRequest;
pub type Pattern = logic::search::hex_pattern::HexPattern;
pub struct RsaPss;
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
    pub unsafe extern "C" fn kernel_rsa_pss_signature_vanity(
        launch: Read Fixed Launch,
        request: Read Fixed Request,
        pattern: Read Fixed Pattern,
        message: Read Slice u8,
        output: ReadWrite Fixed BatchResult,
        records: Write Slice CandidateResult,
    ) {
        unsafe { entry::dispatch::<RsaPss>(launch, request, pattern, message, output, records); }
    } dispatch Grid1d;
}

// SAFETY: the explicit entry delegates the shared six-buffer mechanics to candidate_entry.
unsafe impl Contract for RsaPss {
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
        logic::modes::rsa_pss::rsa_pss(request, payload, counter, pattern)
    }
}
