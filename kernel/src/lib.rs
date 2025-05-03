#![no_std]

extern crate alloc;

mod sha512;
mod edwards25519;
mod precomputed_table;

use bs58;
use cuda_std::address_space; 

fn sha512_compact(input: &[u8]) -> [u8; 64] {
    let mut hasher = crate::sha512::Hash::new();
    hasher.update(input);
    hasher.finalize()
}

fn derrive_public_key_compact(hashed_private_key_bytes: [u8; 64]) -> [u8; 32] {
    use crate::precomputed_table::PRECOMPUTED_TABLE;
    let public_key_bytes = crate::edwards25519::ge_scalarmult(&hashed_private_key_bytes[0..32], &PRECOMPUTED_TABLE).to_bytes();
    public_key_bytes
}

#[cuda_std::kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_vanity_private_key(
    // input
    vanity_prefix_ptr: *const u8, 
    vanity_prefix_len: usize, 
    rng_seed: u64,
    // output
    found_flag_ptr: *mut cuda_std::atomic::AtomicF32,
    found_private_key_ptr: *mut u8,
    found_public_key_ptr: *mut u8,
    found_bs58_encoded_public_key_ptr: *mut u8,
) {
    // read vanity prefix from host
    let vanity_prefix = unsafe { core::slice::from_raw_parts(vanity_prefix_ptr, vanity_prefix_len as usize) };
    
    // initialize rng
    let thread_idx = cuda_std::thread::index() as usize;
    let mut rng_state = thread_idx as u64 ^ rng_seed;

    // generate random input for private key
    const LCG_MULTIPLIER: u64 = 6364136223846793005;
    let mut generate_random_byte = || {
        rng_state = rng_state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(1);
        (rng_state >> 56) as u8
    };
    let mut private_key = [0u8; 32];
    for i in 0..32 {
        private_key[i] = generate_random_byte();
    }
    
    // sha512 hash input
    let mut hashed_private_key_bytes = sha512_compact(&private_key[0..32]);
    
    // apply ed25519 clamping
    hashed_private_key_bytes[0] &= 248;
    hashed_private_key_bytes[31] &= 127;
    hashed_private_key_bytes[31] |= 64;
    
    // ed25519 private key -> public key (first 32 bytes only)
    let public_key_bytes = derrive_public_key_compact(hashed_private_key_bytes);
    
    // bs58 encode public key
    let mut bs58_encoded_public_key = [0u8; 44];
    bs58::encode(&public_key_bytes[0..32]).onto(&mut bs58_encoded_public_key[0..]).unwrap();
    
    // check if public key starts with vanity prefix
    let mut matches = true;
    for i in 0..vanity_prefix_len {
        if bs58_encoded_public_key[i] != unsafe { vanity_prefix[i] } {
            matches = false;
            break;
        }
    }
    
    // if match, copy results to host
    if matches {
        // copy results to host
        let found_flag_slice = unsafe { core::slice::from_raw_parts_mut(found_flag_ptr, 1) };
        let found_private_key = unsafe { core::slice::from_raw_parts_mut(found_private_key_ptr, 32) };
        let found_public_key = unsafe { core::slice::from_raw_parts_mut(found_public_key_ptr, 32) };
        let found_bs58_encoded_public_key = unsafe { core::slice::from_raw_parts_mut(found_bs58_encoded_public_key_ptr, 44) };
        
        let found_flag = &mut found_flag_slice[0];
        found_flag.fetch_add(1.0, core::sync::atomic::Ordering::SeqCst);
        cuda_std::atomic::mid::device_thread_fence(core::sync::atomic::Ordering::SeqCst);
        
        // TODO: need to copy more than 1 single result
        found_private_key.copy_from_slice(&private_key[0..32]);
        found_public_key.copy_from_slice(&public_key_bytes[0..32]);
        found_bs58_encoded_public_key.copy_from_slice(&bs58_encoded_public_key[0..44]);
    }
    cuda_std::thread::sync_threads();
}
