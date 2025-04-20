extern crate alloc;

mod sha;

use cuda_std::prelude::*;
use crate::sha::Hash;
use ed25519_dalek::{SigningKey, VerifyingKey};
use bs58;

/*fn generate_keypair(local_buffer: &[u8; 32]) -> (SigningKey, VerifyingKey, String) {
    // Perform SHA-512 hashing
    /*println!("1");
    let mut hasher = Sha512::new();
    println!("2");
    hasher.update(local_buffer);
    println!("3");
    let mut hash = hasher.finalize();
    println!("4");
    
    // Special ed25519 hash modification
    hash[0] &= 248;
    hash[31] &= 127;
    hash[31] |= 64;*/

    let hash = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];
    
    // Create private key from the modified hash
    println!("1");
    let mut key_bytes = [0u8; 32];
    let mut public_key = [0u8; 32];    
    let mut secret_key = [0u8; 32];    
    /*key_bytes.copy_from_slice(&hash[..32]);
    println!("2");
    let secret_key = SigningKey::from_bytes(&key_bytes);
    println!("3");
    // Derive public key from private key
    let public_key = VerifyingKey::from(&secret_key);
    println!("4");*/
    
    // Base58 encode the public key
    println!("5");
    let encoded_public_key = bs58::encode(&public_key.to_bytes()).into_string();
    println!("6");

    (secret_key, public_key, encoded_public_key)
}*/

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn vecadd(a: &[f32], b: &[f32], c: *mut f32) {
    let mut input = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    // sha512
    let mut hash = Hash::new();
    hash.update(&input[0..32]);
    let mut hashed_input = hash.finalize();
    println!("Hashed input: {:02x?}", hashed_input);

    // Special ed25519 hash modification
    hashed_input[0] &= 248;
    hashed_input[31] = (hashed_input[31] & 127) | 64;

    // ed25519
    let secret_key = SecretKey::from_slice(&hashed_input[0..64]);
    let public_key = secret_key.public_key();
    let public_key = public_key.as_slice();

    // sha512 -> bs58
    let mut bs58_ascii = [0; 64];
    bs58::encode(&public_key[..32]).onto(&mut bs58_ascii[..]).unwrap();
    println!("Encoded hash: {:02x?}", bs58_ascii);

    /*let (secret_key, public_key, encoded_public_key) = generate_keypair(&input);
    println!("Secret key: {:?}", secret_key);
    println!("Public key: {:?}", public_key);
    println!("Encoded public key: {}", encoded_public_key);*/

    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = unsafe { &mut *c.add(idx) };
        *elem = a[idx] + b[idx];
    }
}
