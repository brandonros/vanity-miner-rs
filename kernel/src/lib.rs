#![no_std]

extern crate alloc;

mod sha512;
mod ed25519;
mod ed25519_precomputed_table;
mod base58;

fn sha512_hash(input: &[u8]) -> [u8; 64] {
    use crate::sha512::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(input);
    hasher.finalize()
}

fn ed25519_clamp(hashed_private_key_bytes: &mut [u8]) {
    hashed_private_key_bytes[0] &= 0xF8;
    hashed_private_key_bytes[31] &= 0x7F;
    hashed_private_key_bytes[31] |= 0x40;
}

fn ed25519_derive_public_key(hashed_private_key_bytes: &[u8]) -> [u8; 32] {
    use crate::ed25519_precomputed_table::PRECOMPUTED_TABLE;
    use crate::ed25519::ge_scalarmult;
    let public_key_point = ge_scalarmult(&hashed_private_key_bytes[0..32], &PRECOMPUTED_TABLE);
    let public_key_bytes = public_key_point.to_bytes();
    public_key_bytes
}

fn xorshiro_generate_random_private_key(thread_idx: usize, rng_seed: u64) -> [u8; 32] {
    // Initialize state with thread_idx and seed
    let mut s0 = thread_idx as u64 ^ rng_seed;
    let mut s1 = s0.wrapping_add(0x9E3779B97F4A7C15); // Use golden ratio for second part of state
    
    if s0 == 0 && s1 == 0 {
        s0 = 1; // Avoid all-zero state
    }
    
    let mut private_key = [0u8; 32];
    
    for i in (0..32).step_by(8) {
        // Xoroshiro128** algorithm
        let result = s0.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        
        let bytes = [
            (result >> 56) as u8,
            (result >> 48) as u8,
            (result >> 40) as u8,
            (result >> 32) as u8,
            (result >> 24) as u8,
            (result >> 16) as u8,
            (result >> 8) as u8,
            result as u8
        ];
        
        // State update
        let s1_new = s1 ^ s0;
        s0 = s0.rotate_left(24) ^ s1_new ^ (s1_new << 16);
        s1 = s1_new.rotate_left(37);
        
        let end = core::cmp::min(i + 8, 32);
        private_key[i..end].copy_from_slice(&bytes[0..(end - i)]);
    }
    
    private_key
}

#[cuda_std::kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_vanity_private_key(
    // input
    vanity_prefix_ptr: *const u8, 
    vanity_prefix_len: usize, 
    rng_seed: u64,
    // output
    found_flag_slice_ptr: *mut cuda_std::atomic::AtomicF32,
    found_private_key_ptr: *mut u8,
    found_public_key_ptr: *mut u8,
    found_bs58_encoded_public_key_ptr: *mut u8,
    found_thread_idx_slice_ptr: *mut u32,
) {
    // generate random input for private key from thread index and rng seed
    let thread_idx = cuda_std::thread::index() as usize;
    let private_key = xorshiro_generate_random_private_key(thread_idx, rng_seed);
    
    // sha512 hash private key
    let mut hashed_private_key_bytes = sha512_hash(&private_key[0..32]);
    
    // apply ed25519 clamping to hashed private key
    ed25519_clamp(&mut hashed_private_key_bytes);
    
    // calculate public key from hashed private key with ed25519 point multiplication
    let public_key_bytes = ed25519_derive_public_key(&hashed_private_key_bytes[0..32]);
    
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
        let found_flag_slice = unsafe { core::slice::from_raw_parts_mut(found_flag_slice_ptr, 1) };
        let found_private_key = unsafe { core::slice::from_raw_parts_mut(found_private_key_ptr, 32) };
        let found_public_key = unsafe { core::slice::from_raw_parts_mut(found_public_key_ptr, 32) };
        let found_bs58_encoded_public_key = unsafe { core::slice::from_raw_parts_mut(found_bs58_encoded_public_key_ptr, 64) };
        let found_thread_idx_slice = unsafe { core::slice::from_raw_parts_mut(found_thread_idx_slice_ptr, 1) };
        let found_flag = &mut found_flag_slice[0];
        found_flag.fetch_add(1.0, core::sync::atomic::Ordering::SeqCst);
        cuda_std::atomic::mid::device_thread_fence(core::sync::atomic::Ordering::SeqCst);
        
        // TODO: need to copy more than 1 single result
        found_private_key.copy_from_slice(&private_key[0..32]);
        found_public_key.copy_from_slice(&public_key_bytes[0..32]);
        found_bs58_encoded_public_key.copy_from_slice(&bs58_encoded_public_key[0..64]);
        found_thread_idx_slice[0] = thread_idx as u32;
    }

    // sync threads
    cuda_std::thread::sync_threads();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_hash_correctly() {
        let private_key = [0x4e, 0x89, 0x41, 0xbc, 0xbe, 0x2e, 0x4a, 0x27, 0x55, 0xf2, 0xd3, 0xe3, 0xb9, 0xc4, 0xc6, 0x57, 0xf6, 0x3c, 0x7d, 0xe7, 0x79, 0xc1, 0x34, 0x91, 0x50, 0xc8, 0x09, 0xed, 0x44, 0x3a, 0x8b, 0x8c];

        // sha512
        let mut hashed_private_key_bytes = sha512_hash(&private_key[0..32]);
        let expected = [0x1e, 0x8c, 0x16, 0x6d, 0x45, 0x3b, 0xe1, 0x47, 0xe4, 0x82, 0x39, 0x80, 0xd8, 0x36, 0x98, 0x30, 0x86, 0xc4, 0xc8, 0x1e, 0xec, 0x63, 0x71, 0xb8, 0x00, 0xc1, 0x34, 0x9d, 0xb6, 0x5d, 0xba, 0x4b, 0x1e, 0x0d, 0x01, 0x38, 0x5f, 0x63, 0x7f, 0xed, 0x97, 0x77, 0x61, 0x95, 0x1b, 0xa8, 0x75, 0x45, 0x37, 0x3c, 0x2b, 0x04, 0xb4, 0x31, 0xda, 0x60, 0x82, 0xf5, 0x4c, 0x39, 0x5f, 0xdf, 0x85, 0x10];
        assert_eq!(hashed_private_key_bytes, expected);

        // clamp
        ed25519_clamp(&mut hashed_private_key_bytes);
        assert_eq!(hashed_private_key_bytes[0], 0x18);
        assert_eq!(hashed_private_key_bytes[31], 0x4b);
        
        // derive public key
        let public_key_bytes = ed25519_derive_public_key(&hashed_private_key_bytes[0..32]);
        let expected = [0x0a, 0xf7, 0x64, 0xbe, 0x9b, 0x67, 0x30, 0x71, 0xe9, 0xe3, 0xfd, 0x5e, 0xce, 0xfb, 0x1f, 0x33, 0x73, 0x2f, 0x44, 0xb6, 0x9b, 0x38, 0x5d, 0x0a, 0x94, 0xf6, 0x14, 0x73, 0xbb, 0xb9, 0xf6, 0xf6];
        assert_eq!(public_key_bytes, expected);

        // bs58 encode public key
        let mut bs58_encoded_public_key = [0u8; 64];
        let encoded_len = base58::encode_into_limbs(&public_key_bytes, &mut bs58_encoded_public_key);
        let bs58_encoded_public_key = &bs58_encoded_public_key[0..encoded_len];
        let expected = b"jose1xrZkrKD4sXCvPwyhps7MEQSu7RpsVzPAs5fnc5";
        assert_eq!(bs58_encoded_public_key, expected);
    }
}