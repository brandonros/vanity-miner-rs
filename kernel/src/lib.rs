#![no_std]

extern crate alloc;

use sha2::Digest as _;
use rand_core::{SeedableRng, RngCore};
use rand_xorshift::XorShiftRng;
use bs58;

// change me!
pub const ED25519_COMPACT: bool = false; // true = works, false = IllegalAddress
pub const SHA512_COMPACT: bool = true; // true = works, false = LaunchFailed

fn sha512(input: &[u8]) -> [u8; 64] {
    if SHA512_COMPACT {
        let mut hasher = ed25519_compact::sha512::Hash::new();
        hasher.update(input);
        hasher.finalize()
    } else {
        let mut hasher = sha2::Sha512::new();
        hasher.update(input);
        hasher.finalize().into()
    }
}

fn derrive_public_key(hashed_private_key_bytes: [u8; 64]) -> [u8; 32] {
    if ED25519_COMPACT {
        let mut input = [0u8; 32];
        input.copy_from_slice(&hashed_private_key_bytes[0..32]);
        let public_key_bytes = ed25519_compact::ge_scalarmult_base(&input).to_bytes();
        public_key_bytes
    } else {
        let mut input = [0u8; 32];
        input.copy_from_slice(&hashed_private_key_bytes[0..32]);
        let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(input);
        let point = curve25519_dalek::constants::ED25519_BASEPOINT_TABLE * &scalar;
        let recip = point.Z.invert();
        let x = &point.X * &recip;
        let y = &point.Y * &recip;
        let mut s: [u8; 32];
        s = y.as_bytes();
        let x_bytes = x.as_bytes();
        let x_is_negative = x_bytes[0] & 1;
        s[31] ^= x_is_negative << 7;
        let compressed_point = curve25519_dalek::edwards::CompressedEdwardsY(s);
        let public_key_bytes = compressed_point.to_bytes();
        //public_key_bytes // if you try to return this, IllegalAddress
        [0u8; 32] // if you return this, everything else above it passes?
    }
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
    let vanity_prefix = core::slice::from_raw_parts(vanity_prefix_ptr, vanity_prefix_len as usize);
    cuda_std::thread::sync_threads();
    
    // initialize rng + buffers + hasher + flag
    let idx = cuda_std::thread::index() as usize;
    let mut rng = XorShiftRng::seed_from_u64(rng_seed + idx as u64);
    //let mut rng = Xoroshiro128StarStar::seed_from_u64(rng_seed + idx as u64);
    let mut private_key = [0u8; 32];
    let mut bs58_encoded_public_key = [0u8; 44];
    cuda_std::thread::sync_threads();
    
    // generate random input
    rng.fill_bytes(&mut private_key[0..32]);
    cuda_std::thread::sync_threads();
    
    // sha512 hash input
    let mut hashed_private_key_bytes = sha512(&private_key[0..32]);
    cuda_std::thread::sync_threads();
    
    // apply ed25519 clamping
    hashed_private_key_bytes[0] &= 248;
    hashed_private_key_bytes[31] = (hashed_private_key_bytes[31] & 127) | 64;
    cuda_std::thread::sync_threads();
    
    // ed25519 private key -> public key (first 32 bytes only)
    let public_key_bytes = derrive_public_key(hashed_private_key_bytes);
    cuda_std::thread::sync_threads();
    
    // bs58 encode public key
    bs58::encode(&public_key_bytes[0..32]).onto(&mut bs58_encoded_public_key[0..]).unwrap();
    cuda_std::thread::sync_threads();
    
    // check if public key starts with vanity prefix
    let matches = bs58_encoded_public_key[0..vanity_prefix_len] == *vanity_prefix;
    cuda_std::thread::sync_threads();
    
    // if match, copy results to host
    if matches {
        // copy results to host
        let found_flag_slice = core::slice::from_raw_parts_mut(found_flag_ptr, 1);
        let found_private_key = core::slice::from_raw_parts_mut(found_private_key_ptr, 32);
        let found_public_key = core::slice::from_raw_parts_mut(found_public_key_ptr, 32);
        let found_bs58_encoded_public_key = core::slice::from_raw_parts_mut(found_bs58_encoded_public_key_ptr, 44);
        
        let mut found_flag = &mut found_flag_slice[0];
        found_flag.fetch_add(1.0, core::sync::atomic::Ordering::SeqCst);
        cuda_std::atomic::mid::device_thread_fence(core::sync::atomic::Ordering::SeqCst);
        
        // TODO: need to copy more than 1 single result
        found_private_key.copy_from_slice(&private_key[0..32]);
        found_public_key.copy_from_slice(&public_key_bytes[0..32]);
        found_bs58_encoded_public_key.copy_from_slice(&bs58_encoded_public_key[0..44]);
    }
    cuda_std::thread::sync_threads();
}
