#![no_std]

extern crate alloc;

mod base58;
mod xorshiro;

fn sha512_hash(input: &[u8]) -> [u8; 64] {
    use sha2::{Digest, Sha512};
    let mut hasher = Sha512::new();
    hasher.update(input);
    hasher.finalize().into()
}

fn ed25519_clamp(hashed_private_key_bytes: &mut [u8]) {
    hashed_private_key_bytes[0] &= 248;
    hashed_private_key_bytes[31] &= 63;
    hashed_private_key_bytes[31] |= 64;
}

fn ed25519_derive_public_key(hashed_private_key_bytes: &[u8]) -> [u8; 32] {
    let mut input = [0u8; 32];
    input.copy_from_slice(&hashed_private_key_bytes[0..32]);
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(input);
    let point = curve25519_dalek::constants::ED25519_BASEPOINT_TABLE * &scalar;
    let compressed_point = point.compress();
    let public_key_bytes = compressed_point.to_bytes();
    public_key_bytes
}

#[cuda_std::kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn find_vanity_private_key(
    // input
    vanity_prefix_ptr: *const u8, 
    vanity_prefix_len: usize, 
    vanity_suffix_ptr: *const u8,
    vanity_suffix_len: usize,
    rng_seed: u64,
    // output
    found_matches_slice_ptr: *mut cuda_std::atomic::AtomicF32,
    found_private_key_ptr: *mut u8,
    found_public_key_ptr: *mut u8,
    found_bs58_encoded_public_key_ptr: *mut u8,
    found_thread_idx_slice_ptr: *mut u32,
) {
    // generate random input for private key from thread index and rng seed
    let thread_idx = cuda_std::thread::index() as usize;
    let private_key = xorshiro::generate_random_private_key(thread_idx, rng_seed);
    
    // sha512 hash private key
    let mut hashed_private_key_bytes = sha512_hash(&private_key);

    // take first 32 bytes of hashed private key
    let mut hashed_private_key_bytes = &mut hashed_private_key_bytes[0..32];

    // apply ed25519 clamping to hashed private key
    ed25519_clamp(&mut hashed_private_key_bytes);

    // calculate public key from hashed private key with ed25519 point multiplication
    let public_key_bytes = ed25519_derive_public_key(&hashed_private_key_bytes);

    // bs58 encode public key
    let mut bs58_encoded_public_key = [0u8; 64];
    let encoded_len = base58::encode_into_limbs(&public_key_bytes, &mut bs58_encoded_public_key);

    // check if public key starts with vanity prefix
    let vanity_prefix = unsafe { core::slice::from_raw_parts(vanity_prefix_ptr, vanity_prefix_len as usize) };
    let vanity_suffix = unsafe { core::slice::from_raw_parts(vanity_suffix_ptr, vanity_suffix_len as usize) };
    let mut matches = true;
    for i in 0..vanity_prefix_len {
        if bs58_encoded_public_key[i] != vanity_prefix[i] {
            matches = false;
            break;
        }
    }
    for i in 0..vanity_suffix_len {
        if bs58_encoded_public_key[encoded_len - vanity_suffix_len + i] != vanity_suffix[i] {
            matches = false;
            break;
        }
    }
    
    // if match, copy found match to host
    if matches {
        let found_matches_slice = unsafe { core::slice::from_raw_parts_mut(found_matches_slice_ptr, 1) };
        let found_private_key = unsafe { core::slice::from_raw_parts_mut(found_private_key_ptr, 32) };
        let found_public_key = unsafe { core::slice::from_raw_parts_mut(found_public_key_ptr, 32) };
        let found_bs58_encoded_public_key = unsafe { core::slice::from_raw_parts_mut(found_bs58_encoded_public_key_ptr, 64) };
        let found_thread_idx_slice = unsafe { core::slice::from_raw_parts_mut(found_thread_idx_slice_ptr, 1) };
        let found_matches = &mut found_matches_slice[0];

        // if first find, copy results to host
        if found_matches.load(core::sync::atomic::Ordering::SeqCst) == 0.0 {
            found_private_key.copy_from_slice(&private_key[0..32]);
            found_public_key.copy_from_slice(&public_key_bytes[0..32]);
            found_bs58_encoded_public_key.copy_from_slice(&bs58_encoded_public_key[0..64]);
            found_thread_idx_slice[0] = thread_idx as u32;
        }

        found_matches.fetch_add(1.0, core::sync::atomic::Ordering::SeqCst);
        cuda_std::atomic::mid::device_thread_fence(core::sync::atomic::Ordering::SeqCst);   
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_hash_correctly() {
        let rng_seed = 3314977520786701659;
        let thread_idx = 13604434;
        let private_key = xorshiro::generate_random_private_key(thread_idx, rng_seed);
        let expected = b"\x1A\x3C\x0F\xD9\xC5\xA8\x4E\x8B\xE4\xA9\xD3\x36\x03\x69\xDE\x0C\xCE\x80\x01\x49\xDC\x2E\x5C\x05\xDB\xD5\xB0\xCD\x23\x24\x1B\x0E";
        assert_eq!(private_key, *expected);

        // sha512
        let mut hashed_private_key_bytes = sha512_hash(&private_key[0..32]);
        let expected = b"\x6f\x6d\xc4\xb5\xc8\x7f\x2b\x57\xe0\x27\x04\xbc\x82\x8e\x94\x4d\xbe\x93\x86\xb1\x17\x3f\xa7\x7f\xa6\xad\x9d\x88\xd7\x73\xc8\x49\xc5\x82\x2f\xd4\x62\x45\x4d\x7a\xd5\x98\x77\xb4\xe8\x4c\xd6\x65\x87\x36\x22\x28\x43\x07\x1d\xb8\x54\xbf\x28\xb7\x67\xb9\xa7\x65";
        assert_eq!(hashed_private_key_bytes, *expected);

        // reduce
        let mut hashed_private_key_bytes = &mut hashed_private_key_bytes[0..32];
        let expected = b"\x6f\x6d\xc4\xb5\xc8\x7f\x2b\x57\xe0\x27\x04\xbc\x82\x8e\x94\x4d\xbe\x93\x86\xb1\x17\x3f\xa7\x7f\xa6\xad\x9d\x88\xd7\x73\xc8\x49";
        assert_eq!(hashed_private_key_bytes, *expected);

        // clamp
        ed25519_clamp(&mut hashed_private_key_bytes);
        let expected = b"\x68\x6d\xc4\xb5\xc8\x7f\x2b\x57\xe0\x27\x04\xbc\x82\x8e\x94\x4d\xbe\x93\x86\xb1\x17\x3f\xa7\x7f\xa6\xad\x9d\x88\xd7\x73\xc8\x49";
        assert_eq!(hashed_private_key_bytes, *expected);
        
        // derive public key
        let public_key_bytes = ed25519_derive_public_key(&hashed_private_key_bytes);
        let expected = b"\x0a\xf7\x64\xd3\x34\x40\x71\xf9\x99\xf7\x05\x20\xb3\x1c\xe9\xa4\x52\xf4\xcf\x44\x21\x19\xfe\xa8\x71\x6c\xd7\x55\x85\x11\x86\x30";
        assert_eq!(public_key_bytes, *expected);

        // bs58 encode public key
        let mut bs58_encoded_public_key = [0u8; 64];
        let encoded_len = base58::encode_into_limbs(&public_key_bytes[0..32], &mut bs58_encoded_public_key);
        let bs58_encoded_public_key = &bs58_encoded_public_key[0..encoded_len];
        let expected = b"josexCM7QB9psJfzZtvAzYWAjHb5LSCH594nvCvVSdu";
        assert_eq!(bs58_encoded_public_key, *expected);
    }
}
