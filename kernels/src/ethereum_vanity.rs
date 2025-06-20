/// Handle the infrastructure concerns when a match is found
unsafe fn handle_ethereum_vanity_match_found(
    result: logic::EthereumVanityKeyResult,
    thread_idx: usize,
    found_matches_slice_ptr: *mut cuda_std::atomic::AtomicF32,
    found_private_key_ptr: *mut u8,
    found_public_key_ptr: *mut u8,
    found_address_ptr: *mut u8,
    found_thread_idx_slice_ptr: *mut u32,
) {
    let found_matches_slice = unsafe { core::slice::from_raw_parts_mut(found_matches_slice_ptr, 1) };
    let found_matches = &mut found_matches_slice[0];

    // If first find, copy results to host
    if found_matches.load(core::sync::atomic::Ordering::Relaxed) == 0.0 {
        let found_private_key = unsafe { core::slice::from_raw_parts_mut(found_private_key_ptr, 32) };
        let found_public_key = unsafe { core::slice::from_raw_parts_mut(found_public_key_ptr, 64) };
        let found_address = unsafe { core::slice::from_raw_parts_mut(found_address_ptr, 20) };
        let found_thread_idx_slice = unsafe { core::slice::from_raw_parts_mut(found_thread_idx_slice_ptr, 1) };

        found_private_key.copy_from_slice(&result.private_key);
        found_public_key.copy_from_slice(&result.public_key);
        found_address.copy_from_slice(&result.address);
        found_thread_idx_slice[0] = thread_idx as u32;
    }

    // Increment number of found matches
    found_matches.fetch_add(1.0, core::sync::atomic::Ordering::Relaxed);

    // TODO: do we need device_fence here?
}

#[cuda_std::kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn kernel_find_ethereum_vanity_private_key(
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
    found_address_ptr: *mut u8,
    found_thread_idx_slice_ptr: *mut u32,
) {
    // Prepare request
    let thread_idx = cuda_std::thread::index() as usize;
    let vanity_prefix = unsafe { core::slice::from_raw_parts(vanity_prefix_ptr, vanity_prefix_len) };
    let vanity_suffix = unsafe { core::slice::from_raw_parts(vanity_suffix_ptr, vanity_suffix_len) };
    let request = logic::EthereumVanityKeyRequest {
        prefix: vanity_prefix,
        suffix: vanity_suffix,
        thread_idx,
        rng_seed,
    };
    
    // Call pure business logic
    let result = logic::generate_and_check_ethereum_vanity_key(&request);
    
    // Handle result (adapter layer)
    if result.matches {
        unsafe { 
            handle_ethereum_vanity_match_found(
                result,
                thread_idx,
                found_matches_slice_ptr,
                found_private_key_ptr,
                found_public_key_ptr,
                found_address_ptr,
                found_thread_idx_slice_ptr,
            );
        }
    }
}
