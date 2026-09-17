//! CUDA entry point for p256-public-key: one shared winner per launch.

#![no_std]

extern crate alloc;

use cuda_std::prelude::*;
use logic::{
    modes::p256_public_key::{P256PublicRequest, p256_public},
    search::candidate_result::{BatchResult, CandidateResult},
    search::hex_pattern::HexPattern,
};

/// # Safety
/// Request and pattern pointers must be valid and aligned. `output` must point
/// to one BatchResult initialized to EMPTY before each launch. `message` must
/// hold `message_len` readable bytes when nonzero. Inputs must remain immutable until stream synchronization.
#[kernel]
pub unsafe extern "C" fn kernel_p256_public_key_vanity(
    request: *const P256PublicRequest,
    pattern: *const HexPattern,
    _message: *const u8,
    _message_len: usize,
    start: u64,
    count: u32,
    output: *mut BatchResult,
) {
    let lane = cuda_std::thread::index() as usize;
    if lane >= count as usize {
        return;
    }
    let result = if let Some(counter) = start.checked_add(lane as u64) {
        unsafe { p256_public(&*request, counter, &*pattern) }
    } else {
        CandidateResult::ERROR
    };
    unsafe {
        crate::match_handler::record(lane, result, output);
    }
}

#[path = "../../../common/match_handler.rs"]
mod match_handler;
