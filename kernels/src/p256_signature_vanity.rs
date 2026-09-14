//! CUDA entry point for p256-signature: one candidate per lane.
use cuda_std::prelude::*;
use logic::{
    candidate_result::CandidateResult,
    hex_pattern::HexPattern,
    p256_signature_vanity::{P256SignatureRequest, p256_signature},
};

/// # Safety
/// Request and pattern pointers must be valid and aligned. `results` must hold
/// `count` writable records; `message` must hold `message_len` readable bytes
/// when nonzero. Inputs must remain immutable until stream synchronization.
#[kernel]
pub unsafe extern "C" fn kernel_p256_signature_vanity(
    request: *const P256SignatureRequest,
    pattern: *const HexPattern,
    message: *const u8,
    message_len: usize,
    start: u64,
    count: u32,
    results: *mut CandidateResult,
) {
    let lane = crate::utilities::get_thread_idx();
    if lane >= count as usize {
        return;
    }
    let result = if let Some(counter) = start.checked_add(lane as u64) {
        let message = if message_len == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(message, message_len) }
        };
        unsafe { p256_signature(&*request, message, counter, &*pattern) }
    } else {
        CandidateResult::ERROR
    };
    unsafe {
        results.add(lane as usize).write(result);
    }
}
