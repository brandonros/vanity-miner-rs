//! CUDA entry point for p256-public-key: one candidate per lane.
use cuda_std::prelude::*;
use logic::{
    candidate_result::CandidateResult,
    hex_pattern::HexPattern,
    p256_public_key_vanity::{P256PublicRequest, p256_public},
};

/// # Safety
/// Request and pattern pointers must be valid and aligned. `results` must hold
/// `count` writable records; `message` must hold `message_len` readable bytes
/// when nonzero. Inputs must remain immutable until stream synchronization.
#[kernel]
pub unsafe extern "C" fn kernel_p256_public_key_vanity(
    request: *const P256PublicRequest,
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
        let _message = if message_len == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(message, message_len) }
        };
        unsafe { p256_public(&*request, counter, &*pattern) }
    } else {
        CandidateResult::ERROR
    };
    unsafe {
        results.add(lane as usize).write(result);
    }
}
