//! CUDA entry point for rsa-modulus: one shared winner per launch.
use cuda_std::prelude::*;
use logic::{
    modes::rsa_modulus::{RsaModulusRequest, rsa_modulus},
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

use logic::modes::rsa_modulus::pipeline::{self, Counts, Pair, SearchConfig, Task};
use core::sync::atomic::Ordering;

unsafe fn increment(ptr: *mut u32) -> u32 {
    unsafe { cuda_std::atomic::mid::atomic_fetch_add_u32_device(ptr, Ordering::Relaxed, 1) }
}

/// # Safety
/// Config is immutable; tasks has `capacity` initialized slots. Counts is zeroed
/// before this launch. All stages execute in order on the same stream.
#[kernel]
pub unsafe extern "C" fn kernel_rsa_generate_v3(
    config: *const SearchConfig, tasks: *mut Task, start: u64, capacity: u32, counts: *mut Counts,
) {
    let lane = cuda_std::thread::index() as u32;
    if lane >= capacity { return; }
    let task = unsafe { &mut *tasks.add(lane as usize) };
    if task.state != 0 { return; }
    unsafe { increment(core::ptr::addr_of_mut!((*counts).p_tested)); }
    let Some(id) = start.checked_add(u64::from(lane)) else {
        unsafe { increment(core::ptr::addr_of_mut!((*counts).errors)); }
        return;
    };
    let Some(p) = pipeline::generate_p(unsafe { &*config }, id) else {
        unsafe { increment(core::ptr::addr_of_mut!((*counts).errors)); }
        return;
    };
    if pipeline::range_first(unsafe { &*config }) {
        task.p = p;
        task.id = id;
        task.state = 3;
    } else if pipeline::probable_p(&p) {
        task.p = p;
        task.id = id;
        task.state = 1;
        unsafe { increment(core::ptr::addr_of_mut!((*counts).p_accepted)); }
    }
}

/// # Safety
/// Active and winners each have `capacity` slots. The preceding generate stage
/// has completed. No q stage may read these buffers concurrently.
#[kernel]
pub unsafe extern "C" fn kernel_rsa_ranges_v3(
    config: *const SearchConfig, tasks: *mut Task, active: *mut u32,
    winners: *mut u32, capacity: u32, counts: *mut Counts,
) {
    let lane = cuda_std::thread::index() as u32;
    if lane >= capacity { return; }
    let task = unsafe { &mut *tasks.add(lane as usize) };
    let unchecked = task.state == 3;
    if task.state == 1 || unchecked {
        match pipeline::prepare_range(unsafe { &*config }, task) {
            Ok(true) => {
                if unchecked && !pipeline::probable_p(&task.p) {
                    *task = Task::EMPTY;
                } else {
                    if unchecked { unsafe { increment(core::ptr::addr_of_mut!((*counts).p_accepted)); } }
                    unsafe { increment(core::ptr::addr_of_mut!((*counts).ranges)); }
                }
            }
            Ok(false) => {}
            Err(_) => { unsafe { increment(core::ptr::addr_of_mut!((*counts).errors)); } }
        }
    }
    if task.state == 2 {
        let index = unsafe { increment(core::ptr::addr_of_mut!((*counts).active)) };
        if index < capacity {
            unsafe { *active.add(index as usize) = lane; *winners.add(lane as usize) = 0; }
        } else {
            unsafe { increment(core::ptr::addr_of_mut!((*counts).errors)); }
        }
    }
}

/// # Safety
/// Range construction has completed. Tasks and active indices are immutable for
/// this entire launch; winner counters are separate allocations, accessed only
/// atomically. Pairs has `capacity` slots, enough for one result per active task.
#[kernel]
pub unsafe extern "C" fn kernel_rsa_search_v3(
    config: *const SearchConfig, pattern: *const HexPattern, tasks: *const Task,
    active: *const u32, winners: *mut u32, pairs: *mut Pair, capacity: u32, counts: *mut Counts,
) {
    let lane = cuda_std::thread::index() as u32;
    if lane >= capacity { return; }
    let active_count = unsafe { (*counts).active };
    if active_count == 0 || active_count > capacity { return; }
    let slot = unsafe { *active.add((lane % active_count) as usize) };
    if slot >= capacity {
        unsafe { increment(core::ptr::addr_of_mut!((*counts).errors)); }
        return;
    }
    let task = unsafe { &*tasks.add(slot as usize) };
    let Some(q) = pipeline::q_at(unsafe { &*config }, task, lane / active_count) else { return; };
    unsafe { increment(core::ptr::addr_of_mut!((*counts).q_tested)); }
    if pipeline::eligible_pair(&task.p, &q, unsafe { &*pattern })
        && unsafe { increment(winners.add(slot as usize)) } == 0
    {
        let index = unsafe { increment(core::ptr::addr_of_mut!((*counts).matches)) };
        if index < capacity {
            unsafe { *pairs.add(index as usize) = Pair { p: task.p, q, id: task.id }; }
        } else {
            unsafe { increment(core::ptr::addr_of_mut!((*counts).errors)); }
        }
    }
}

/// # Safety
/// The search stage has completed. Only this stage mutates active task cursors.
/// Winning factors are erased and retired before any replacement work starts.
#[kernel]
pub unsafe extern "C" fn kernel_rsa_advance_v3(
    tasks: *mut Task, active: *const u32, winners: *const u32, capacity: u32, counts: *const Counts,
) {
    let index = cuda_std::thread::index() as u32;
    let active_count = unsafe { (*counts).active };
    if active_count == 0 || active_count > capacity || index >= active_count { return; }
    let slot = unsafe { *active.add(index as usize) };
    if slot >= capacity { return; }
    let task = unsafe { &mut *tasks.add(slot as usize) };
    task.winner = unsafe { *winners.add(slot as usize) };
    let assigned = (capacity - 1 - index) / active_count + 1;
    pipeline::finish_tile(task, assigned);
}
