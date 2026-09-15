//! Structured candidate entry for bitcoin.
use cuda_std::prelude::*;
use logic::search::{
    candidate_result::{BatchResult, CandidateResult},
    xoroshiro::BatchSeed,
};

/// # Safety
/// Request and pattern are valid aligned records. Message is readable for its
/// length. Output is initialized to EMPTY; inputs remain alive through synchronization.
#[kernel]
pub unsafe extern "C" fn kernel_bitcoin_vanity(
    request: *const BatchSeed,
    pattern: *const logic::search::vanity::BytePattern,
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
    let result = match start.checked_add(lane as u64) {
        Some(counter) => unsafe { logic::modes::bitcoin::candidate(&*request, counter, &*pattern) },
        None => CandidateResult::ERROR,
    };
    unsafe {
        crate::match_handler::record(lane, result, output);
    }
}
