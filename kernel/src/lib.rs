extern crate alloc;

use cuda_std::prelude::*;
use ed25519_compact::{PublicKey, SecretKey, ge_scalarmult_base};
use hmac_sha512::Hash;
use gpu_rand::xoroshiro::Xoroshiro128StarStar;
use gpu_rand::xoroshiro::rand_core::SeedableRng;
use gpu_rand::xoroshiro::rand_core::RngCore;
use bs58;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd(a: &[f32], b: &[f32], c: *mut f32) {
    // generate random input
    let mut rng_seed = 42;
    let num_iterations = 1;

    for _ in 0..num_iterations {
        // generate random input for seed
        println!("generating random input");
        let mut rng = Xoroshiro128StarStar::seed_from_u64(rng_seed);
        let mut input = [0u8; 32];
        rng.fill_bytes(&mut input);
        println!("generated random input");

        // sha512 hash input
        println!("hashing input");
        let mut hash = Hash::new();
        hash.update(&input[0..32]);
        let mut hashed_input = hash.finalize();
        println!("hashed input");

        // Apply Ed25519 clamping
        println!("clamping input");
        hashed_input[0] &= 248;
        hashed_input[31] = (hashed_input[31] & 127) | 64;
        println!("clamped input");

        // Now use the library's ge_scalarmult_base to compute the public key
        // from the first 32 bytes of your hashed and clamped input
        // Use the library's function to compute public key from scalar
        println!("computing public key");
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&hashed_input[0..32]);
        let public_key_bytes = ge_scalarmult_base(&scalar).to_bytes();
        println!("computed public key");

        // bs58 encode public key
        println!("encoding public key");
        let mut bs58_encoded_public_key = [0; 64];
        bs58::encode(&public_key_bytes[..32]).onto(&mut bs58_encoded_public_key[..]).unwrap();
        println!("encoded public key");

        // check if public key starts with 31313131
        if bs58_encoded_public_key[0] == 0x31 && bs58_encoded_public_key[1] == 0x31 {
            println!("found something");

            // build a secret key
            /*let public_key = PublicKey::new(public_key_bytes);
            let mut combined_key = [0u8; 64];
            combined_key[0..32].copy_from_slice(&input); // Original input as seed
            combined_key[32..64].copy_from_slice(&public_key_bytes);
            let secret_key = SecretKey::new(combined_key);*/

            println!("input: {:02x?}", input);
            println!("Public key: {:02x?}", public_key_bytes);
            println!("Base58 encoded public key: {:02x?}", bs58_encoded_public_key);
            break;
        } else {
            println!("no match");
        }

        rng_seed += 1;
    }
}
