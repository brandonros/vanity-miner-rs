//! CUDA entry point for p256-signature: one shared winner per launch.

#![no_std]
#![feature(abi_ptx)]

use kernel_common::{lane, record};
use logic::{
    modes::p256_signature::{P256SignatureRequest, p256_signature},
    search::candidate_result::{BatchResult, CandidateResult},
    search::hex_pattern::HexPattern,
};

/// # Safety
/// Request and pattern pointers must be valid and aligned. `output` must point
/// to one BatchResult initialized to EMPTY before each launch. `message` must
/// hold `message_len` readable bytes when nonzero. Inputs must remain immutable until stream synchronization.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn kernel_p256_signature_vanity(
    request: *const P256SignatureRequest,
    pattern: *const HexPattern,
    message: *const u8,
    message_len: usize,
    start: u64,
    count: u32,
    output: *mut BatchResult,
) {
    let lane = lane();
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
        record(lane, result, output);
    }
}
