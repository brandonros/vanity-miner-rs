//! Device runtime shared by the kernels: heap, panic handler, thread index,
//! and atomic publication of one structured candidate result per launch.

#![no_std]
#![feature(stdarch_nvptx)]

use core::alloc::{GlobalAlloc, Layout};
use core::arch::nvptx;
use core::sync::atomic::{AtomicU32, Ordering};
use logic::search::candidate_result::{BatchResult, CandidateResult};

// CUDA provides these system calls to every PTX module.
unsafe extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

struct DeviceHeap;

unsafe impl GlobalAlloc for DeviceHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Device malloc returns 16-byte aligned blocks.
        if layout.align() > 16 {
            return core::ptr::null_mut();
        }
        unsafe { malloc(layout.size()) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { free(ptr) }
    }
}

#[global_allocator]
static HEAP: DeviceHeap = DeviceHeap;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe { nvptx::trap() }
}

/// Global index of this thread in a one-dimensional launch.
pub fn lane() -> usize {
    unsafe {
        let block = nvptx::_block_idx_x() as u32 * nvptx::_block_dim_x() as u32;
        (block + nvptx::_thread_idx_x() as u32) as usize
    }
}

/// # Safety
/// Output is valid and initialized to EMPTY, and the host reads only after synchronization.
pub unsafe fn record(lane: usize, result: CandidateResult, output: *mut BatchResult) {
    match result.status {
        CandidateResult::STATUS_MISS => {}
        CandidateResult::STATUS_MATCH => unsafe {
            let matches = AtomicU32::from_ptr(&raw mut (*output).matches);
            if matches.fetch_add(1, Ordering::Relaxed) == 0 {
                (*output).candidate = result;
                (*output).lane = lane as u32;
            }
        },
        _ => unsafe {
            AtomicU32::from_ptr(&raw mut (*output).errors).fetch_add(1, Ordering::Relaxed);
        },
    }
}
