//! Shared device mechanics. Mode entries retain explicit signatures and names.
use logic::search::{candidate_abi::{Contract, Launch}, candidate_result::{BatchResult, CandidateResult}};
unsafe extern "C" {
    #[link_name = "llvm_metal.linear_thread_index"]
    fn thread_index() -> u32;
    #[link_name = "llvm_metal.atomic_add_device_u32"]
    fn atomic_add(pointer: *mut u32, value: u32) -> u32;
}
/// # Safety
/// Six aligned, disjoint buffers matching C::layout(). Payload has message_len
/// readable bytes (one allocated byte if empty); audit has count records if enabled.
/// Output is initialized to EMPTY. No host access until synchronous completion.
pub unsafe fn dispatch<C: Contract>(launch: *const Launch, request: *const C::Request,
    pattern: *const C::Pattern, message: *const u8, output: *mut BatchResult,
    records: *mut CandidateResult) {
    unsafe {
        let launch = &*launch;
        let lane = thread_index();
        if lane >= launch.count { return; }
        let payload = core::slice::from_raw_parts(message, launch.message_len as usize);
        let result = match launch.counter(lane) {
            Some(counter) => C::candidate(&*request, &*pattern, payload, counter),
            None => CandidateResult::ERROR,
        };
        if launch.audit != 0 { records.add(lane as usize).write(result); }
        match result.status {
            CandidateResult::STATUS_MISS => {},
            CandidateResult::STATUS_MATCH => {
                if atomic_add(core::ptr::addr_of_mut!((*output).matches), 1) == 0 {
                    (*output).candidate = result;
                    (*output).lane = lane;
                }
            },
            _ => { atomic_add(core::ptr::addr_of_mut!((*output).errors), 1); }
        }
    }
}
