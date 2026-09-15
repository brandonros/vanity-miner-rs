//! Structured candidate entry for shallenge.
use cuda_std::prelude::*;
use logic::search::{
    candidate_result::{BatchResult, CandidateResult},
    xoroshiro::BatchSeed,
};

/// # Safety
/// Request and pattern are valid aligned records. Message is readable for its
/// length. Output is initialized to EMPTY; inputs remain alive through synchronization.
#[kernel]
pub unsafe extern "C" fn kernel_shallenge(
    request: *const BatchSeed,
    pattern: *const [u8; 32],
    message: *const u8,
    message_len: usize,
    start: u64,
    count: u32,
    output: *mut BatchResult,
) {
    let lane = cuda_std::thread::index() as usize;
    if lane >= count as usize {
        return;
    }
    let result = match start.checked_add(lane as u64) {
        Some(counter) => unsafe {
            logic::modes::shallenge::candidate(
                &*request,
                counter,
                &*pattern,
                if message_len == 0 {
                    &[]
                } else {
                    core::slice::from_raw_parts(message, message_len)
                },
            )
        },
        None => CandidateResult::ERROR,
    };
    unsafe {
        crate::match_handler::record(lane, result, output);
    }
}
