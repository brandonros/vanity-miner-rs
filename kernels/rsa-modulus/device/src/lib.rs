//! One resumable RSA miner. Each lane exclusively owns its persistent task.

#![no_std]

extern crate alloc;

use core::sync::atomic::Ordering;
use cuda_std::prelude::*;
use logic::modes::rsa_modulus::{self as mining, Counts, Pair, SearchConfig, Task};
use logic::search::hex_pattern::HexPattern;

unsafe fn add(ptr: *mut u32, value: u32) -> u32 {
    unsafe { cuda_std::atomic::mid::atomic_fetch_add_u32_device(ptr, Ordering::Relaxed, value) }
}

/// # Safety
/// Config and pattern are immutable. Tasks contains `capacity` initialized slots,
/// exclusively owned by their lanes. Pairs has `capacity` slots, and counts is
/// zeroed before launch. Allocations do not alias and remain alive until completion.
/// No concurrent launch may use this workspace. Host reads only after synchronization.
#[kernel]
pub unsafe extern "C" fn kernel_rsa_modulus_vanity(
    config: *const SearchConfig,
    pattern: *const HexPattern,
    tasks: *mut Task,
    pairs: *mut Pair,
    start: u64,
    capacity: u32,
    steps: u32,
    counts: *mut Counts,
) {
    let lane = cuda_std::thread::index() as u32;
    if lane >= capacity {
        return;
    }
    let Some(id) = start.checked_add(u64::from(lane)) else {
        unsafe {
            add(core::ptr::addr_of_mut!((*counts).errors), 1);
        }
        return;
    };
    let (local, pair) = unsafe {
        mining::mine(
            &*config,
            &*pattern,
            &mut *tasks.add(lane as usize),
            id,
            capacity,
            steps,
        )
    };
    // Publish at most one result per lane. Aggregate statistics once per lane,
    // keeping shared atomics out of the inner factor/q loop.
    if let Some(pair) = pair {
        let index = unsafe { add(core::ptr::addr_of_mut!((*counts).matches), 1) };
        if index < capacity {
            unsafe {
                *pairs.add(index as usize) = pair;
            }
        } else {
            unsafe {
                add(core::ptr::addr_of_mut!((*counts).errors), 1);
            }
        }
    }
    macro_rules! publish {
        ($($field:ident),+) => { $(
            if local.$field != 0 {
                unsafe { add(core::ptr::addr_of_mut!((*counts).$field), local.$field); }
            }
        )+ };
    }
    publish!(p_tested, p_accepted, ranges, q_tested, active, errors);
}
