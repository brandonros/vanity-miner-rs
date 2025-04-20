extern crate alloc;

use cuda_std::prelude::*;
use hmac_sha512::Hash;
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
        let mut rng = DefaultRand::seed_from_u64(rng_seed);
        let mut input = [0u8; 32];
        rng.fill_bytes(&mut input);

        // sha512
        let mut hash = Hash::new();
        hash.update(&input[0..32]);
        let mut hashed_input = hash.finalize();

        // Special ed25519 hash modification
        hashed_input[0] &= 248;
        hashed_input[31] = (hashed_input[31] & 127) | 64;

        // ed25519
        let seed = Seed::from_slice(&hashed_input[0..32]).unwrap();
        let key_pair = KeyPair::from_seed(seed);
        let secret_key = key_pair.sk;
        let public_key = key_pair.pk;

        // bs58
        let mut bs58_encoded_public_key = [0; 64];
        bs58::encode(&public_key[..32]).onto(&mut bs58_encoded_public_key[..]).unwrap();

        if bs58_encoded_public_key[0] == 0x31 && bs58_encoded_public_key[1] == 0x31 && 
            bs58_encoded_public_key[2] == 0x31 && bs58_encoded_public_key[3] == 0x31 {
            println!("input: {:02x?}", input);
            println!("SHA-512 Hashed input: {:02x?}", hashed_input);
            println!("Public key: {:02x?}", public_key);
            println!("Base58 encoded public key: {:02x?}", bs58_encoded_public_key);
        }

        rng_seed += 1;
    }

    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = unsafe { &mut *c.add(idx) };
        *elem = a[idx] + b[idx];
    }
}
