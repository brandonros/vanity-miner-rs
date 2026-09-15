//! Atomic publication of one structured candidate result per launch.
use core::sync::atomic::Ordering;
use logic::search::candidate_result::{BatchResult, CandidateResult};

/// Output is valid and initialized to EMPTY, and the host reads only after synchronization.
pub unsafe fn record(lane: usize, result: CandidateResult, output: *mut BatchResult) {
    match result.status {
        CandidateResult::STATUS_MISS => {}
        CandidateResult::STATUS_MATCH => unsafe {
            if cuda_std::atomic::mid::atomic_fetch_add_u32_device(
                core::ptr::addr_of_mut!((*output).matches),
                Ordering::Relaxed,
                1,
            ) == 0
            {
                (*output).candidate = result;
                (*output).lane = lane as u32;
            }
        },
        _ => unsafe {
            cuda_std::atomic::mid::atomic_fetch_add_u32_device(
                core::ptr::addr_of_mut!((*output).errors),
                Ordering::Relaxed,
                1,
            );
        },
    }
}
