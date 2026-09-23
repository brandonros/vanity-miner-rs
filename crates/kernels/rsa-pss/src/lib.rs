//! CUDA entry point for rsa-pss: one shared winner per launch.

#![cfg(target_arch = "nvptx64")]
#![no_std]
#![feature(abi_ptx)]

use kernel_common::{lane, record};
use logic::{
    modes::rsa_pss::{RsaPssRequest, rsa_pss},
    search::candidate_result::{BatchResult, CandidateResult},
    search::hex_pattern::HexPattern,
};

/// # Safety
/// Request and pattern pointers must be valid and aligned. `output` must point
/// to one BatchResult initialized to EMPTY before each launch. `message` must
/// hold `message_len` readable bytes when nonzero. Inputs must remain immutable until stream synchronization.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn kernel_rsa_pss_signature_vanity(
    request: *const RsaPssRequest,
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
        unsafe { rsa_pss(&*request, message, counter, &*pattern) }
    } else {
        CandidateResult::ERROR
    };
    unsafe {
        record(lane, result, output);
    }
}
