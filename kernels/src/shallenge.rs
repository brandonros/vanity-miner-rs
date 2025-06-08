/// Handle the infrastructure concerns when a better hash is found
unsafe fn handle_shallenge_match_found(
    result: logic::ShallengeResult,
    thread_idx: usize,
    found_matches_slice_ptr: *mut core::sync::atomic::AtomicU32,
    found_hash_ptr: *mut u8,
    found_nonce_ptr: *mut u8,
    found_nonce_len_ptr: *mut usize,
    found_thread_idx_slice_ptr: *mut u32,
) {
    let found_matches_slice = unsafe { core::slice::from_raw_parts_mut(found_matches_slice_ptr, 1) };
    let found_matches = &mut found_matches_slice[0];

    // Always copy the better result (race condition is acceptable here)
    let found_hash = unsafe { core::slice::from_raw_parts_mut(found_hash_ptr, 32) };
    let found_nonce = unsafe { core::slice::from_raw_parts_mut(found_nonce_ptr, 64) };
    let found_nonce_len = unsafe { core::slice::from_raw_parts_mut(found_nonce_len_ptr, 1) };
    let found_thread_idx_slice = unsafe { core::slice::from_raw_parts_mut(found_thread_idx_slice_ptr, 1) };

    found_hash.copy_from_slice(&result.hash);
    found_nonce.copy_from_slice(&result.nonce);
    found_nonce_len[0] = result.nonce_len;
    found_thread_idx_slice[0] = thread_idx as u32;

    // Increment number of found matches
    found_matches.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

#[cuda_std::kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_better_shallenge_nonce(
    // input
    username_ptr: *const u8,
    username_len: usize,
    target_hash_ptr: *const u8,
    rng_seed: u64,
    // output
    found_matches_slice_ptr: *mut core::sync::atomic::AtomicU32,
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
