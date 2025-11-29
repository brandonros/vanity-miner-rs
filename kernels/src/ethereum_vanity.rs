use crate::utilities;

// TODO: kernel
#[unsafe(no_mangle)]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe extern "C" fn kernel_find_ethereum_vanity_private_key(
    // input
    vanity_prefix_ptr: *const u8, 
    vanity_prefix_len: usize, 
    vanity_suffix_ptr: *const u8,
    vanity_suffix_len: usize,
    rng_seed: u64,
    // output
    found_matches_slice_ptr: *mut u32,
    found_private_key_ptr: *mut u8,
    found_public_key_ptr: *mut u8,
    found_address_ptr: *mut u8,
    found_thread_idx_slice_ptr: *mut u32,
) {
    // Prepare request
    let thread_idx = utilities::get_thread_idx();
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
        handle_match! {
            thread_idx: thread_idx,
            found_matches_ptr: found_matches_slice_ptr,
            copies: [
                result.private_key => found_private_key_ptr, 32;
                result.public_key => found_public_key_ptr, 64;
                result.address => found_address_ptr, 20;
            ],
            found_thread_idx_ptr: found_thread_idx_slice_ptr,
        }
    }
}
