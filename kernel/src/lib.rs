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
    // input
    vanity_prefix_ptr: *const u8, 
    vanity_prefix_len: usize, 
    rng_seed: u64,
    // output
    found_flag_ptr: *mut f32,
    found_private_key_ptr: *mut u8,
    found_public_key_ptr: *mut u8,
    found_bs58_encoded_public_key_ptr: *mut u8,
) {
    // read vanity prefix from host
    let vanity_prefix = core::slice::from_raw_parts(vanity_prefix_ptr, vanity_prefix_len as usize);
    cuda_std::thread::sync_threads();

    // initialize rng + buffers + hasher + flag
    let idx = cuda_std::thread::index() as usize;
    let mut rng = XorShiftRng::seed_from_u64(rng_seed + idx as u64);
    let mut private_key = [0u8; 32];
    let mut bs58_encoded_public_key = [0u8; 44];
    let mut hasher = Hash::new();
    cuda_std::thread::sync_threads();

   // generate random input
   rng.fill_bytes(&mut private_key[0..32]);
   cuda_std::thread::sync_threads();

   // sha512 hash input
   hasher.update(&private_key[0..32]);
   let mut hashed_private_key = hasher.finalize();
   cuda_std::thread::sync_threads();

   // apply ed25519 clamping
   hashed_private_key[0] &= 248;
   hashed_private_key[31] = (hashed_private_key[31] & 127) | 64;
   cuda_std::thread::sync_threads();

   // ed25519 private key -> public key (first 32 bytes only)
   let public_key_bytes = ge_scalarmult_base(&hashed_private_key[0..32]).to_bytes();
   cuda_std::thread::sync_threads();

   // bs58 encode public key
   bs58::encode(&public_key_bytes[0..32]).onto(&mut bs58_encoded_public_key[0..]).unwrap();
   cuda_std::thread::sync_threads();

   // check if public key starts with vanity prefix
   if bs58_encoded_public_key[0..vanity_prefix_len] == *vanity_prefix {
       // copy results to host
       let found_flag_slice = core::slice::from_raw_parts_mut(found_flag_ptr, 1);
       let found_private_key = core::slice::from_raw_parts_mut(found_private_key_ptr, 32);
       let found_public_key = core::slice::from_raw_parts_mut(found_public_key_ptr, 32);
       let found_bs58_encoded_public_key = core::slice::from_raw_parts_mut(found_bs58_encoded_public_key_ptr, 44);

       // TODO: this needs to be atomic
       let mut found_flag = &mut found_flag_slice[0];
       *found_flag += 1.0;

       // TODO: need to copy more than 1 single result
       found_private_key.copy_from_slice(&private_key[0..32]);
       found_public_key.copy_from_slice(&public_key_bytes[0..32]);
       found_bs58_encoded_public_key.copy_from_slice(&bs58_encoded_public_key[0..44]);
   }
   cuda_std::thread::sync_threads();
}
