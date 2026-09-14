use cuda_std::prelude::*;

/// Handle the infrastructure concerns when a better hash is found
unsafe fn handle_shallenge_match_found(
    result: logic::ShallengeResult,
    thread_idx: usize,
    found_matches_slice_ptr: *mut u32,
    found_hash_ptr: *mut u8,
    found_nonce_ptr: *mut u8,
    found_nonce_len_ptr: *mut usize,
    found_thread_idx_slice_ptr: *mut u32,
) {
    // Keep the first improvement to atomically claim the slot, not necessarily
    // the best hash in this launch. Later improvements are counted but discarded.
    // This preserves a consistent hash/nonce pair without a minimum reduction.
    handle_match! {
        thread_idx: thread_idx,
        found_matches_ptr: found_matches_slice_ptr,
        copies: [
            result.hash => found_hash_ptr, 32;
            result.nonce => found_nonce_ptr, 64;
            scalar: result.nonce_len => found_nonce_len_ptr;
        ],
        found_thread_idx_ptr: found_thread_idx_slice_ptr,
    }
}

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_find_better_shallenge_nonce(
    // input
    username_ptr: *const u8,
    username_len: usize,
    target_hash_ptr: *const u8,
    rng_seed: u64,
    // output
    found_matches_slice_ptr: *mut u32,
    found_hash_ptr: *mut u8,
    found_nonce_ptr: *mut u8,
    found_nonce_len_ptr: *mut usize,
    found_thread_idx_slice_ptr: *mut u32,
) {
    // Prepare request
    let thread_idx = cuda_std::thread::index() as usize;
    let username = unsafe { core::slice::from_raw_parts(username_ptr, username_len) };
    let target_hash_slice = unsafe { core::slice::from_raw_parts(target_hash_ptr, 32) };
    let target_hash: &[u8; 32] = unsafe { &*(target_hash_slice.as_ptr() as *const [u8; 32]) };
    
    let request = logic::ShallengeRequest {
        username,
        username_len,
        target_hash,
        thread_idx,
        rng_seed,
    };
    
    // Call pure business logic
    let result = logic::generate_and_check_shallenge(&request);
    
    // Handle result (adapter layer)
    if result.is_better {
        unsafe { 
            handle_shallenge_match_found(
                result,
                thread_idx,
                found_matches_slice_ptr,
                found_hash_ptr,
                found_nonce_ptr,
                found_nonce_len_ptr,
                found_thread_idx_slice_ptr,
            );
        }
    }
}
