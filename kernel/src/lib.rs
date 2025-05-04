#![no_std]

extern crate alloc;

mod sha512;
mod ed25519;
mod ed25519_precomputed_table;
mod base58;

const LCG_MULTIPLIER: u64 = 6364136223846793005;

fn sha512_compact(input: &[u8]) -> [u8; 64] {
    use crate::sha512::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(input);
    hasher.finalize()
}

fn derrive_public_key_compact(hashed_private_key_bytes: [u8; 64]) -> [u8; 32] {
    use crate::ed25519_precomputed_table::PRECOMPUTED_TABLE;
    use crate::ed25519::ge_scalarmult;
    let public_key_point = ge_scalarmult(&hashed_private_key_bytes[0..32], &PRECOMPUTED_TABLE);
    let public_key_bytes = public_key_point.to_bytes();
    public_key_bytes
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
    // generate random input for private key from thread index and rng seed
    let thread_idx = cuda_std::thread::index() as usize;
    let mut rng_state = thread_idx as u64 ^ rng_seed;
    let mut generate_random_bytes = || {
        rng_state = rng_state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(1);
        [
            (rng_state >> 56) as u8,
            (rng_state >> 48) as u8,
            (rng_state >> 40) as u8,
            (rng_state >> 32) as u8,
            (rng_state >> 24) as u8,
            (rng_state >> 16) as u8,
            (rng_state >> 8) as u8,
            rng_state as u8
        ]
    };
    let mut private_key = [0u8; 32];
    for i in (0..32).step_by(8) {
        let bytes = generate_random_bytes();
        let end = core::cmp::min(i + 8, 32);
        private_key[i..end].copy_from_slice(&bytes[0..(end - i)]);
    }
    
    // sha512 hash private key
    let mut hashed_private_key_bytes = sha512_compact(&private_key[0..32]);
    
    // apply ed25519 clamping to hashed private key
    hashed_private_key_bytes[0] &= 248;
    hashed_private_key_bytes[31] &= 127;
    hashed_private_key_bytes[31] |= 64;
    
    // calculate public key from hashed private key with ed25519 point multiplication
    let public_key_bytes = derrive_public_key_compact(&hashed_private_key_bytes[0..32]);
    
    // bs58 encode public key
    let mut bs58_encoded_public_key = [0u8; 64];
    let _encoded_len = base58::encode_into_limbs(&public_key_bytes, &mut bs58_encoded_public_key);

    // read vanity prefix from host
    let vanity_prefix = unsafe { core::slice::from_raw_parts(vanity_prefix_ptr, vanity_prefix_len as usize) };

    // check if public key starts with vanity prefix
    let mut matches = true;
    for i in 0..vanity_prefix_len {
        if bs58_encoded_public_key[i] != vanity_prefix[i] {
            matches = false;
            break;
        }
    }
    
    // if match, copy found match to host
    if matches {
        let found_flag_slice = unsafe { core::slice::from_raw_parts_mut(found_flag_ptr, 1) };
        let found_private_key = unsafe { core::slice::from_raw_parts_mut(found_private_key_ptr, 32) };
        let found_public_key = unsafe { core::slice::from_raw_parts_mut(found_public_key_ptr, 32) };
        let found_bs58_encoded_public_key = unsafe { core::slice::from_raw_parts_mut(found_bs58_encoded_public_key_ptr, 64) };
        
        let found_flag = &mut found_flag_slice[0];
        found_flag.fetch_add(1.0, core::sync::atomic::Ordering::SeqCst);
        cuda_std::atomic::mid::device_thread_fence(core::sync::atomic::Ordering::SeqCst);
        
        // TODO: need to copy more than 1 single result
        found_private_key.copy_from_slice(&private_key[0..32]);
        found_public_key.copy_from_slice(&public_key_bytes[0..32]);
        found_bs58_encoded_public_key.copy_from_slice(&bs58_encoded_public_key[0..64]);
    }

    // sync threads
    cuda_std::thread::sync_threads();
}
