//! Stock-Rust Bitcoin device entry.
#![no_std]
mod contract;
pub use contract::*;
#[cfg(target_arch = "nvptx64")]
use logic::search::candidate_result::CandidateResult;

#[cfg(target_arch = "nvptx64")]
unsafe extern "C" {
    #[link_name = "llvm_metal.linear_thread_index"]
    fn thread_index() -> u32;
    #[link_name = "llvm_metal.atomic_add_device_u32"]
    fn atomic_add(pointer: *mut u32, value: u32) -> u32;
}

/// # Safety
/// Disjoint initialized Launch, writable BatchResult initialized to EMPTY,
/// and count writable CandidateResults if audit != 0 (one otherwise).
/// All records are naturally aligned; host readers wait for GPU completion.
#[cfg(target_arch = "nvptx64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kernel_bitcoin_vanity(
    launch: *const Launch,
    output: *mut logic::search::candidate_result::BatchResult,
    records: *mut CandidateResult,
) {
    unsafe {
        let launch = &*launch;
        let lane = thread_index();
        if lane >= launch.count {
            return;
        }
        let result = candidate(launch, lane);
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
