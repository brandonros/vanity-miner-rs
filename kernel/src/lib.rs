#![no_std]

extern crate alloc;

mod sha512;
mod base58;
mod xorshiro;
mod ed25519;

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
    let hashed_private_key_bytes = sha512::sha512_32bytes_from_bytes(&private_key);

    // take first 32 bytes of hashed private key
    let mut hashed_private_key_32 = [0u8; 32];
    hashed_private_key_32.copy_from_slice(&hashed_private_key_bytes[0..32]);

    // calculate public key from hashed private key with ed25519 point multiplication
    let public_key_bytes = ed25519::ed25519_derive_public_key(&hashed_private_key_32);

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

        // increment number of found matches
        found_matches.fetch_add(1.0, core::sync::atomic::Ordering::SeqCst);
        cuda_std::atomic::mid::device_thread_fence(core::sync::atomic::Ordering::SeqCst);   
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_hash_correctly_387357874134630424_2755802() {
        // xorshio
        let rng_seed = 387357874134630424;
        let thread_idx = 2755802;
        let private_key = xorshiro::generate_random_private_key(thread_idx, rng_seed);
        let expected = hex::decode("2dddc675f7dc652066be3101c58d79bfdbc535784fa67ea7acb35861a2f30a0a").unwrap();
        assert_eq!(private_key, *expected);

        // sha512
        let hashed_private_key_bytes = sha512::sha512_32bytes_from_bytes(&private_key);
        let expected = hex::decode("1a710ef882e2967ee517fdea799e9fb6a508530ed16757d0a29086170d4f0d8d9e23b3a2e634100cac9021b8bd8c401a1fdc2de0e0070af4c57fa0a546a16b4b").unwrap();
        assert_eq!(hashed_private_key_bytes, *expected);

        // reduce
        let mut hashed_private_key_32 = [0u8; 32];
        hashed_private_key_32.copy_from_slice(&hashed_private_key_bytes[0..32]);
        let expected = hex::decode("1a710ef882e2967ee517fdea799e9fb6a508530ed16757d0a29086170d4f0d8d").unwrap();
        assert_eq!(hashed_private_key_32, *expected);

        // derive public key
        let public_key_bytes = ed25519::ed25519_derive_public_key(&hashed_private_key_32);
        let expected = hex::decode("625e8c3953df5b05609358fcaff3fee9df0c572d4834a1c1975604cea3107be5").unwrap();
        assert_eq!(public_key_bytes, *expected);

        // bs58 encode public key
        let mut bs58_encoded_public_key = [0u8; 64];
        let encoded_len = base58::encode_into_limbs(&public_key_bytes[0..32], &mut bs58_encoded_public_key);
        let bs58_encoded_public_key = &bs58_encoded_public_key[0..encoded_len];
        let expected = hex::decode("37637a61454366754b363832614634464452737a5a3661396e3439416871766e7643746948597259484c4265").unwrap();
        assert_eq!(*bs58_encoded_public_key, *expected);

        // utf8
        use alloc::string::String;
        let bs58_encoded_public_key_string = String::from_utf8(bs58_encoded_public_key.to_vec()).unwrap();
        let expected = "7czaECfuK682aF4FDRszZ6a9n49AhqvnvCtiHYrYHLBe";
        assert_eq!(bs58_encoded_public_key_string, expected);
    }
}
