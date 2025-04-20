#![no_std]

extern crate alloc;

use ed25519_compact::ge_scalarmult_base;
use ed25519_compact::sha512::Hash;
use rand_core::{SeedableRng, RngCore};
use rand_xorshift::XorShiftRng;
use bs58;

#[cuda_std::kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_vanity_private_key(
    vanity_prefix_ptr: *const u8, 
    vanity_prefix_len: usize, 
    rng_seed: u64,
    max_num_iterations: usize,
    found_flag_ptr: *mut f32,
    found_private_key_ptr: *mut u8,
    found_public_key_ptr: *mut u8,
    found_bs58_encoded_public_key_ptr: *mut u8,
) {
    // read slices from host
    let found_flag_slice = core::slice::from_raw_parts_mut(found_flag_ptr, 1);
    let mut found_flag = &mut found_flag_slice[0];
    let vanity_prefix = core::slice::from_raw_parts(vanity_prefix_ptr, vanity_prefix_len as usize);
    let found_private_key = core::slice::from_raw_parts_mut(found_private_key_ptr, 32);
    let found_public_key = core::slice::from_raw_parts_mut(found_public_key_ptr, 32);
    let found_bs58_encoded_public_key = core::slice::from_raw_parts_mut(found_bs58_encoded_public_key_ptr, 44);

    // initialize rng + buffers + hasher
    let idx = cuda_std::thread::index() as usize;
    let mut rng = XorShiftRng::seed_from_u64(rng_seed + idx as u64);
    let mut private_key = [0u8; 32];
    let mut bs58_encoded_public_key = [0u8; 44];
    let mut hasher = Hash::new();

    // loop until match is found
    for _ in 0..max_num_iterations {
        // check if match has been found in another thread
        if *found_flag != 0.0 {
            return;
        }

        // generate random input
        rng.fill_bytes(&mut private_key[0..32]);

        // sha512 hash input
        hasher.reset();
        hasher.update(&private_key[0..32]);
        let mut hashed_private_key = hasher.finalize();

        // apply ed25519 clamping
        hashed_private_key[0] &= 248;
        hashed_private_key[31] = (hashed_private_key[31] & 127) | 64;

        // ed25519 private key -> public key (first 32 bytes only)
        let public_key_bytes = ge_scalarmult_base(&hashed_private_key[0..32]).to_bytes();

        // bs58 encode public key
        bs58::encode(&public_key_bytes[0..32]).onto(&mut bs58_encoded_public_key[0..]).unwrap();

        // check if public key starts with vanity prefix
        if bs58_encoded_public_key[0..vanity_prefix_len] == *vanity_prefix {
            // copy results to host
            found_private_key.copy_from_slice(&private_key[0..32]);
            found_public_key.copy_from_slice(&public_key_bytes[0..32]);
            found_bs58_encoded_public_key.copy_from_slice(&bs58_encoded_public_key[0..44]);

            // Then set the flag and use system_fence to ensure visibility across blocks
            *found_flag = 1.0;
            return;
        }
    }
}
