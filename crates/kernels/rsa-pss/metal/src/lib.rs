//! Stock-Rust RSA-PSS device entry; one verified CRT operation per candidate.
#![no_std]
mod contract;
pub use contract::*;
#[cfg(target_arch = "nvptx64")]
use logic::{
    modes::rsa_pss::RsaPssRequest,
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
/// All six buffers are disjoint and naturally aligned. Launch, request, pattern
/// and message_len readable message bytes (one allocated byte for empty) are initialized and immutable. Output
/// is initialized to BatchResult::EMPTY. Records holds count writable candidates
/// when audit != 0 (one otherwise). Host readers wait for GPU completion.
#[cfg(target_arch = "nvptx64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kernel_rsa_pss_signature_vanity(
    launch: *const Launch,
    request: *const RsaPssRequest,
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
