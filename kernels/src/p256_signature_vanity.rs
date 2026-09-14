//! CUDA entry point for p256-signature: one shared winner per launch.
use cuda_std::prelude::*;
use logic::{
    candidate_result::{BatchResult, CandidateResult},
    hex_pattern::HexPattern,
    p256_signature_vanity::{P256SignatureRequest, p256_signature},
};

/// # Safety
/// Request and pattern pointers must be valid and aligned. `output` must point
/// to one BatchResult initialized to EMPTY before each launch. `message` must
/// hold `message_len` readable bytes when nonzero. Inputs must remain immutable until stream synchronization.
#[kernel]
pub unsafe extern "C" fn kernel_p256_signature_vanity(
    request: *const P256SignatureRequest,
    pattern: *const HexPattern,
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
    match result.status {
        0 => {}
        1 => {
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
