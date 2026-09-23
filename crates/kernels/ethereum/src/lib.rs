//! Structured candidate entry for ethereum.

#![cfg(target_arch = "nvptx64")]
#![no_std]
#![feature(abi_ptx)]

use kernel_common::{lane, record};
use logic::search::{
    candidate_result::{BatchResult, CandidateResult},
    xoroshiro::BatchSeed,
};

/// # Safety
/// Request and pattern are valid aligned records. Message is readable for its
/// length. Output is initialized to EMPTY; inputs remain alive through synchronization.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn kernel_ethereum_vanity(
    request: *const BatchSeed,
    pattern: *const logic::search::vanity::BytePattern,
    _message: *const u8,
    _message_len: usize,
    start: u64,
    count: u32,
    output: *mut BatchResult,
) {
    let lane = lane();
    if lane >= count as usize {
        return;
    }
    let result = match start.checked_add(lane as u64) {
        Some(counter) => unsafe {
            logic::modes::ethereum::candidate(&*request, counter, &*pattern)
        },
        None => CandidateResult::ERROR,
    };
    unsafe {
        record(lane, result, output);
    }
}
