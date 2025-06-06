#![no_std]

extern crate alloc;

/// Handle the infrastructure concerns when a match is found
unsafe fn handle_vanity_match_found(
    result: logic::VanityKeyResult,
    thread_idx: usize,
    found_matches_slice_ptr: *mut cuda_std::atomic::AtomicF32,
    found_private_key_ptr: *mut u8,
    found_public_key_ptr: *mut u8,
    found_bs58_encoded_public_key_ptr: *mut u8,
    found_thread_idx_slice_ptr: *mut u32,
) {
    let found_matches_slice = unsafe { core::slice::from_raw_parts_mut(found_matches_slice_ptr, 1) };
    let found_matches = &mut found_matches_slice[0];

    // If first find, copy results to host
    if found_matches.load(core::sync::atomic::Ordering::SeqCst) == 0.0 {
        let found_private_key = unsafe { core::slice::from_raw_parts_mut(found_private_key_ptr, 32) };
        let found_public_key = unsafe { core::slice::from_raw_parts_mut(found_public_key_ptr, 32) };
        let found_bs58_encoded_public_key = unsafe { core::slice::from_raw_parts_mut(found_bs58_encoded_public_key_ptr, 64) };
        let found_thread_idx_slice = unsafe { core::slice::from_raw_parts_mut(found_thread_idx_slice_ptr, 1) };

        found_private_key.copy_from_slice(&result.private_key);
        found_public_key.copy_from_slice(&result.public_key);
        found_bs58_encoded_public_key.copy_from_slice(&result.encoded_public_key);
        found_thread_idx_slice[0] = thread_idx as u32;
    }

    // Increment number of found matches
    found_matches.fetch_add(1.0, core::sync::atomic::Ordering::SeqCst);
    cuda_std::atomic::mid::device_thread_fence(core::sync::atomic::Ordering::SeqCst);
}

#[cuda_std::kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_vanity_private_key(
    // input
    vanity_prefix_ptr: *const u8, 
    vanity_prefix_len: usize, 
    vanity_suffix_ptr: *const u8,
    vanity_suffix_len: usize,
    rng_seed: u64,
    // output
    found_matches_slice_ptr: *mut cuda_std::atomic::AtomicF32,
    found_private_key_ptr: *mut u8,
    found_public_key_ptr: *mut u8,
    found_bs58_encoded_public_key_ptr: *mut u8,
    found_thread_idx_slice_ptr: *mut u32,
) {
    // Prepare request
    let thread_idx = cuda_std::thread::index() as usize;
    let vanity_prefix = unsafe { core::slice::from_raw_parts(vanity_prefix_ptr, vanity_prefix_len) };
    let vanity_suffix = unsafe { core::slice::from_raw_parts(vanity_suffix_ptr, vanity_suffix_len) };
    let request = logic::VanityKeyRequest {
        prefix: vanity_prefix,
        suffix: vanity_suffix,
        thread_idx,
        rng_seed,
    };
    
    // Call pure business logic
    let result = logic::generate_and_check_vanity_key(&request);
    
    // Handle result (adapter layer)
    if result.matches {
        unsafe { 
            handle_vanity_match_found(
                result,
                thread_idx,
                found_matches_slice_ptr,
                found_private_key_ptr,
                found_public_key_ptr,
                found_bs58_encoded_public_key_ptr,
                found_thread_idx_slice_ptr,
            );
        }
    }
}

/// Handle the infrastructure concerns when a better hash is found
unsafe fn handle_shallenge_match_found(
    result: logic::ShallengeResult,
    thread_idx: usize,
    found_matches_slice_ptr: *mut cuda_std::atomic::AtomicF32,
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
    found_matches.fetch_add(1.0, core::sync::atomic::Ordering::SeqCst);
    cuda_std::atomic::mid::device_thread_fence(core::sync::atomic::Ordering::SeqCst);
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
    found_matches_slice_ptr: *mut cuda_std::atomic::AtomicF32,
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
