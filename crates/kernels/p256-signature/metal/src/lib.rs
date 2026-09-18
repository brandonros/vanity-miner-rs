//! Stock-Rust p256-signature device entry.
#![no_std]
mod contract;
pub use contract::*;
#[cfg(target_arch = "nvptx64")]
use logic::{
    modes::p256_signature::P256SignatureRequest,
    search::{
        candidate_result::{BatchResult, CandidateResult},
        hex_pattern::HexPattern,
    },
};
#[cfg(target_arch = "nvptx64")]
unsafe extern "C" {
    #[link_name = "llvm_metal.linear_thread_index"]
    fn thread_index() -> u32;
    #[link_name = "llvm_metal.atomic_add_device_u32"]
    fn atomic_add(pointer: *mut u32, value: u32) -> u32;
}
/// # Safety
/// Naturally aligned, disjoint initialized records; message contains message_len
/// bytes (one allocated byte for empty). Output starts at BatchResult::EMPTY.
/// Records has count writable entries if audit != 0, one otherwise.
/// Host readers wait for completion; count and counter range are validated.
#[cfg(target_arch = "nvptx64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kernel_p256_signature_vanity(
    launch: *const Launch,
    request: *const P256SignatureRequest,
    pattern: *const HexPattern,
    message: *const u8,
    output: *mut BatchResult,
    records: *mut CandidateResult,
) {
    unsafe {
        let launch = &*launch;
        let lane = thread_index();
        if lane >= launch.count {
            return;
        }
        let message = core::slice::from_raw_parts(message, launch.message_len as usize);
        let result = candidate(launch, &*request, &*pattern, message, lane);
        if launch.audit != 0 {
            records.add(lane as usize).write(result);
        }
        match result.status {
            CandidateResult::STATUS_MISS => {}
            CandidateResult::STATUS_MATCH => {
                if atomic_add(core::ptr::addr_of_mut!((*output).matches), 1) == 0 {
                    (*output).candidate = result;
                    (*output).lane = lane;
                }
            }
            _ => {
                atomic_add(core::ptr::addr_of_mut!((*output).errors), 1);
            }
        }
    }
}
