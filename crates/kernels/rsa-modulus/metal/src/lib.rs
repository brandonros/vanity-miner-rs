//! Stock-Rust resumable RSA miner; one persistent task belongs to each lane.
#![no_std]
mod contract;
pub use contract::*;
#[cfg(target_arch = "nvptx64")]
use logic::{
    modes::rsa_modulus::{self as mining, Counts, Pair, SearchConfig, Task},
    search::hex_pattern::HexPattern,
};
#[cfg(target_arch = "nvptx64")]
unsafe extern "C" {
    #[link_name = "llvm_metal.linear_thread_index"]
    fn thread_index() -> u32;
    #[link_name = "llvm_metal.atomic_add_device_u32"]
    fn atomic_add(p: *mut u32, n: u32) -> u32;
}
/// # Safety
/// Disjoint initialized Launch/config/pattern; capacity writable Tasks and Pairs;
/// zeroed Counts. Each lane owns its task. Host waits for completion before reads
/// or the next launch. Allocation sizes and launch work are checked by the host.
#[cfg(target_arch = "nvptx64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kernel_rsa_modulus_vanity(
    launch: *const Launch,
    config: *const SearchConfig,
    pattern: *const HexPattern,
    tasks: *mut Task,
    pairs: *mut Pair,
    counts: *mut Counts,
) {
    unsafe {
        let launch = &*launch;
        let lane = thread_index();
        if lane >= launch.capacity {
            return;
        }
        let Some(id) = launch.start.checked_add(u64::from(lane)) else {
            atomic_add(core::ptr::addr_of_mut!((*counts).errors), 1);
            return;
        };
        // Keep the algorithm's volatile zeroization in private memory. Publish the
        // resulting persistent record only once after the owning lane's work ends.
        let mut task = tasks.add(lane as usize).read();
        let (local, pair) = mining::mine(
            &*config,
            &*pattern,
            &mut task,
            id,
            launch.capacity,
            launch.steps,
        );
        tasks.add(lane as usize).write(task);
        if let Some(pair) = pair {
            let index = atomic_add(core::ptr::addr_of_mut!((*counts).matches), 1);
            if index < launch.capacity {
                pairs.add(index as usize).write(pair);
            } else {
                atomic_add(core::ptr::addr_of_mut!((*counts).errors), 1);
            }
        }
        macro_rules! publish {($($field:ident),+)=>{$(if local.$field!=0 {atomic_add(core::ptr::addr_of_mut!((*counts).$field),local.$field);})+};}
        publish!(p_tested, p_accepted, ranges, q_tested, active, errors);
    }
}
