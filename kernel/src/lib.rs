#![no_std]

extern crate alloc;

/// Handle the infrastructure concerns when a match is found
unsafe fn handle_match_found(
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
            handle_match_found(
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_hash_correctly() {
        // sha512
        let private_key: [u8; 32] = hex::decode("61a314b0183724ea0e5f237584cb76092e253b99783d846a5b10db155128eafd").unwrap().try_into().unwrap();
        let hashed_private_key_bytes = logic::sha512_32bytes_from_bytes(&private_key);
        let expected = hex::decode("152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb0f5b8817a8e43510f83333417178f2f59fdc3c723199303a5f9be71af2f7b664").unwrap();
        assert_eq!(hashed_private_key_bytes, *expected);

        // derive public key
        let public_key_bytes = logic::ed25519_derive_public_key(&hashed_private_key_bytes);
        let expected = hex::decode("0af764c1b6133a3a0abd7ef9c853791b687ce1e235f9dc8466d886da314dbea7").unwrap();
        assert_eq!(public_key_bytes, *expected);

        // bs58 encode public key
        let mut bs58_encoded_public_key = [0u8; 64];
        let encoded_len = logic::base58_encode(&public_key_bytes, &mut bs58_encoded_public_key);
        let bs58_encoded_public_key = &bs58_encoded_public_key[0..encoded_len];
        let expected = hex::decode("6a6f7365413875757746426a58707558423879453233437845756d596758336a486251677753627166504c").unwrap();
        assert_eq!(*bs58_encoded_public_key, *expected);

        // utf8
        use alloc::string::String;
        let bs58_encoded_public_key_string = String::from_utf8(bs58_encoded_public_key.to_vec()).unwrap();
        let expected = "joseA8uuwFBjXpuXB8yE23CxEumYgX3jHbQgwSbqfPL";
        assert_eq!(bs58_encoded_public_key_string, expected);
    }
}
