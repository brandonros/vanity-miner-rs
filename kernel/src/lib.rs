#![no_std]

extern crate alloc;

use ed25519_compact::ge_scalarmult_base;
use ed25519_compact::sha512::Hash;
use rand_xoshiro::rand_core::{RngCore as _, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use bs58;

#[cuda_std::kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_vanity_private_key(vanity_prefix_ptr: *const u8, vanity_prefix_len: usize, rng_seed: u64) {
    let vanity_prefix = core::slice::from_raw_parts(vanity_prefix_ptr, vanity_prefix_len as usize);

    let idx = cuda_std::thread::index_1d() as usize;
    let mut rng = Xoshiro256StarStar::seed_from_u64(rng_seed + idx as u64);
    let mut hasher = Hash::new();
    let mut private_key = [0u8; 32];
    let mut bs58_encoded_public_key = [0u8; 44];
    let mut num_iterations = 0;

    loop {
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
            /*cuda_std::println!("found match");
            cuda_std::println!("Private key: {:02x?}", private_key);
            cuda_std::println!("Public key: {:02x?}", public_key_bytes);
            cuda_std::println!("Base58 encoded public key: {:02x?}", bs58_encoded_public_key);*/
            break;
        }

        num_iterations += 1;
    }
}
