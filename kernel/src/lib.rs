extern crate alloc;

mod mock_rng;

use cuda_std::prelude::*;
use ed25519_compact::{PublicKey, SecretKey, ge_scalarmult_base};
use hmac_sha512::Hash;
//use gpu_rand::xoroshiro::Xoroshiro128StarStar;
use crate::mock_rng::MockXoroshiro128StarStar; 
use gpu_rand::xoroshiro::rand_core::SeedableRng;
use gpu_rand::xoroshiro::rand_core::RngCore;
use bs58;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd(a: &[f32], b: &[f32], c: *mut f32) {
    let mut rng_seed: u64 = 42; // TODO: make this unique for each thread?
    let mut num_iterations: u64 = 0;

    loop {
        // generate random input for seed
        let mut rng = MockXoroshiro128StarStar::seed_from_u64(rng_seed);
        let mut input = [0u8; 32];
        rng.fill_bytes(&mut input);

        // sha512 hash input
        let mut hash = Hash::new();
        hash.update(&input[0..32]);
        let mut hashed_input = hash.finalize();

        // Apply Ed25519 clamping
        hashed_input[0] &= 248;
        hashed_input[31] = (hashed_input[31] & 127) | 64;

        // Now use the library's ge_scalarmult_base to compute the public key
        // from the first 32 bytes of your hashed and clamped input
        // Use the library's function to compute public key from scalar
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&hashed_input[0..32]);
        let public_key_bytes = ge_scalarmult_base(&scalar).to_bytes();

        // bs58 encode public key
        let mut bs58_encoded_public_key = [0; 64];
        bs58::encode(&public_key_bytes[..32]).onto(&mut bs58_encoded_public_key[..]).unwrap();

        // check if public key starts with a
        if bs58_encoded_public_key[0] == 0x61 {
            println!("found something");

            // build a secret key
            /*let public_key = PublicKey::new(public_key_bytes);
            let mut combined_key = [0u8; 64];
            combined_key[0..32].copy_from_slice(&input); // Original input as seed
            combined_key[32..64].copy_from_slice(&public_key_bytes);
            let secret_key = SecretKey::new(combined_key);*/

            /*println!("input: {:02x?}", input);
            println!("Public key: {:02x?}", public_key_bytes);
            println!("Base58 encoded public key: {:02x?}", bs58_encoded_public_key);*/
            break;
        }

        num_iterations += 1;

        if num_iterations % 1000 == 0 {
            println!("num_iterations: {}", num_iterations);
        }

        rng_seed += 1;
    }
}
