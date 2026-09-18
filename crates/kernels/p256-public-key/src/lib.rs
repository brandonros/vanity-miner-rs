#![no_std]
pub mod contract;
pub use contract::*;
#[cfg(target_arch = "nvptx64")]
#[path = "../../common/candidate_entry.rs"]
mod entry;
#[cfg(target_arch = "nvptx64")]
use logic::search::{
    candidate_abi::Launch,
    candidate_result::{BatchResult, CandidateResult},
};
/// # Safety
/// Buffers satisfy the shared candidate ABI, including dynamic payload/audit spans.
#[cfg(target_arch = "nvptx64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kernel_p256_public_key_vanity(
    launch: *const Launch,
    request: *const Request,
    pattern: *const Pattern,
    message: *const u8,
    output: *mut BatchResult,
    records: *mut CandidateResult,
) {
    unsafe {
        entry::dispatch::<P256Public>(launch, request, pattern, message, output, records);
    }
}
