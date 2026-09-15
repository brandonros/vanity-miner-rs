//! CUDA entry point for rsa-modulus: one shared winner per launch.
use cuda_std::prelude::*;
use logic::{
    modes::rsa_modulus_vanity::{RsaModulusRequest, rsa_modulus},
    search::candidate_result::{BatchResult, CandidateResult},
    search::hex_pattern::HexPattern,
};

/// # Safety
/// Request and pattern pointers must be valid and aligned. `output` must point
/// to one BatchResult initialized to EMPTY before each launch. `message` must
/// hold `message_len` readable bytes when nonzero. Inputs must remain immutable until stream synchronization.
#[kernel]
pub unsafe extern "C" fn kernel_rsa_modulus_vanity_v2(
    request: *const RsaModulusRequest,
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
        unsafe { rsa_modulus(&*request, counter, &*pattern) }
    } else {
        CandidateResult::ERROR
    };
    match result.status {
        CandidateResult::STATUS_MISS => {}
        CandidateResult::STATUS_MATCH => {
            handle_match! {
                thread_idx: lane,
                found_matches_ptr: core::ptr::addr_of_mut!((*output).matches),
                copies: [scalar: result => core::ptr::addr_of_mut!((*output).candidate);],
                found_thread_idx_ptr: core::ptr::addr_of_mut!((*output).lane),
            }
        }
        _ => unsafe {
            cuda_std::atomic::mid::atomic_fetch_add_u32_device(
                core::ptr::addr_of_mut!((*output).errors),
                core::sync::atomic::Ordering::Relaxed,
                1,
            );
        },
    }
}
