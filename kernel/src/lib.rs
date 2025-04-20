extern crate alloc;

use cuda_std::prelude::*;
use hmac_sha512::Hash;
use ed25519_compact::SecretKey;
use bs58;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd(a: &[f32], b: &[f32], c: *mut f32) {
    let mut input = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    println!("input: {:02x?}", input);

    // sha512
    let mut hash = Hash::new();
    hash.update(&input[0..32]);
    let mut hashed_input = hash.finalize();
    println!("SHA-512 Hashed input: {:02x?}", hashed_input);

    // Special ed25519 hash modification
    hashed_input[0] &= 248;
    hashed_input[31] = (hashed_input[31] & 127) | 64;

    // ed25519
    let secret_key = SecretKey::from_slice(&hashed_input[0..64]).unwrap();
    let public_key = secret_key.public_key();
    let public_key = public_key.as_slice();
    println!("Public key: {:02x?}", public_key);

    // bs58
    let mut bs58_encoded_public_key = [0; 64];
    bs58::encode(&public_key[..32]).onto(&mut bs58_encoded_public_key[..]).unwrap();
    println!("Base58 encoded public key: {:02x?}", bs58_encoded_public_key);

    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = unsafe { &mut *c.add(idx) };
        *elem = a[idx] + b[idx];
    }
}
