extern crate alloc;

use cuda_std::prelude::*;
use ed25519_compact::{KeyPair, Seed};
use gpu_rand::DefaultRand;
use gpu_rand::xoroshiro::rand_core::SeedableRng;
use gpu_rand::xoroshiro::rand_core::RngCore;
use bs58;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd(a: &[f32], b: &[f32], c: *mut f32) {
    // generate random input
    let mut rng_seed = 0;

    loop {
        // generate random input for seed
        let mut rng = DefaultRand::seed_from_u64(rng_seed);
        let mut input = [0u8; 32];
        rng.fill_bytes(&mut input);

        // pass random input to seed to derive ed25519 key pair
        let seed = Seed::from_slice(&input).unwrap();
        let key_pair = KeyPair::from_seed(seed);
        let secret_key = key_pair.sk;
        let public_key = key_pair.pk;

        // bs58 encode public key
        let mut bs58_encoded_public_key = [0; 64];
        bs58::encode(&public_key[..32]).onto(&mut bs58_encoded_public_key[..]).unwrap();

        // check if public key starts with 31313131
        if bs58_encoded_public_key[0] == 0x31 && bs58_encoded_public_key[1] == 0x31 && 
            bs58_encoded_public_key[2] == 0x31 && bs58_encoded_public_key[3] == 0x31 {
            println!("input: {:02x?}", input);
            println!("Public key: {:02x?}", public_key);
            println!("Base58 encoded public key: {:02x?}", bs58_encoded_public_key);
            break;
        }

        rng_seed += 1;
    }
}
