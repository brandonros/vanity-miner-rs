use seq_macro::seq;

use crate::sha256;
use crate::xoroshiro;

pub struct ShallengeRequest<'a> {
    pub username: &'a [u8],
    pub username_len: usize,
    pub target_hash: &'a [u8; 32],
    pub thread_idx: usize,
    pub rng_seed: u64,
}

pub struct ShallengeResult {
    pub hash: [u8; 32],
    pub nonce: [u8; 64], // Max nonce size
    pub nonce_len: usize,
    pub is_better: bool,
}

pub fn compare_hashes(a: &[u8; 32], b: &[u8; 32]) -> i32 {
    seq!(I in 0..32 {
        if a[I] < b[I] { return -1; }
        if a[I] > b[I] { return 1; }
    });
    0
}

pub fn shallenge(username: &[u8], username_len: usize, nonce: &[u8], nonce_len: usize) -> [u8; 32] {
    let mut input = [0u8; 32];
    let mut pos = 0;
    
    // Copy username
    input[pos..pos + username_len].copy_from_slice(&username[..username_len]);
    pos += username_len;
    
    // Add separator '/'
    input[pos] = b'/';
    pos += 1;
    
    // Copy nonce
    input[pos..pos + nonce_len].copy_from_slice(&nonce[..nonce_len]);
    
    // Hash only the used portion
    sha256::sha256_32_from_bytes(&input)
}

pub fn test_nonce(
    thread_idx: usize,
    rng_seed: u64,
    username: &[u8], 
    username_len: usize, 
    target_hash: &[u8; 32]
) -> i32 {
    let mut nonce = [0u8; 21];
    xoroshiro::generate_base64_nonce(thread_idx, rng_seed, &mut nonce);
    let hash = shallenge(username, username_len, &nonce, 21);
    compare_hashes(&hash, target_hash)
}

pub fn generate_and_check_shallenge(request: &ShallengeRequest) -> ShallengeResult {
    let mut nonce = [0u8; 21]; // Fixed nonce size for now
    xoroshiro::generate_base64_nonce(request.thread_idx, request.rng_seed, &mut nonce);
    
    let hash = shallenge(request.username, request.username_len, &nonce, 21);
    let is_better = compare_hashes(&hash, request.target_hash) < 0;
    
    let mut result_nonce = [0u8; 64];
    result_nonce[..21].copy_from_slice(&nonce);
    
    ShallengeResult {
        hash,
        nonce: result_nonce,
        nonce_len: 21,
        is_better,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shallenge() {
        let username: [u8; 10] = "brandonros".as_bytes().try_into().unwrap();
        let nonce: [u8; 21] = "000000000000000000000".as_bytes().try_into().unwrap();
        let result = shallenge(&username, 10, &nonce, 21);
        let expected: [u8; 32] = hex::decode("f7a41dae1196282f0a544a8c7f1bbf61bda79307dc424c0d9febd27b08e1bf78").unwrap().try_into().unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compare_hashes() {
        let better: [u8; 32] = hex::decode("0000000000000004340F267BA07B90AED63F69DA590F155C140E7CD9786D65DE").unwrap().try_into().unwrap();
        let worse: [u8; 32] = hex::decode("0000000000000038D5CDE5593531FD567B5F15562811C50FC2A45E5F2A458A65").unwrap().try_into().unwrap();
        assert_eq!(compare_hashes(&better, &worse), -1);
        assert_eq!(compare_hashes(&worse, &better), 1);
        assert_eq!(compare_hashes(&better, &better), 0);
    }

    // Capture KAT: derive the hash by running the pipeline once, then bake the
    // value in. The (thread_idx, rng_seed) tuple is arbitrary — the point is
    // regression coverage for the end-to-end seed→nonce→hash path.
    fn shallenge_kat_inputs() -> (usize, u64, [u8; 10]) {
        (0, 12345u64, *b"brandonros")
    }

    #[test]
    fn test_test_nonce_returns_comparison_against_target() {
        let (thread_idx, rng_seed, username) = shallenge_kat_inputs();
        let expected_hash: [u8; 32] = hex::decode("c3750f8711bf809f46de1f01eceb6f4e6fde670ad8a3e2a600a0e0b7357654c9").unwrap().try_into().unwrap();

        // Equal target → 0
        assert_eq!(test_nonce(thread_idx, rng_seed, &username, 10, &expected_hash), 0);
        // Target all-1s (max) → derived hash is smaller → -1
        let max = [0xffu8; 32];
        assert_eq!(test_nonce(thread_idx, rng_seed, &username, 10, &max), -1);
        // Target all-0s (min) → derived hash is greater → 1
        let zero = [0u8; 32];
        assert_eq!(test_nonce(thread_idx, rng_seed, &username, 10, &zero), 1);
    }

    #[test]
    fn test_generate_and_check_shallenge_end_to_end() {
        let (thread_idx, rng_seed, username) = shallenge_kat_inputs();
        let expected_hash: [u8; 32] = hex::decode("c3750f8711bf809f46de1f01eceb6f4e6fde670ad8a3e2a600a0e0b7357654c9").unwrap().try_into().unwrap();
        let max = [0xffu8; 32];

        let request = ShallengeRequest {
            username: &username,
            username_len: 10,
            target_hash: &max,
            thread_idx,
            rng_seed,
        };
        let result = generate_and_check_shallenge(&request);
        assert_eq!(result.hash, expected_hash);
        assert_eq!(result.nonce_len, 21);
        // every nonce byte must lie in the base64 alphabet
        let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for &b in &result.nonce[..result.nonce_len] {
            assert!(alphabet.contains(&b), "nonce byte {:#x} not in alphabet", b);
        }
        assert_eq!(result.is_better, true, "hash < max should be better");

        // With target = all-0s, derived hash is greater → not better
        let zero = [0u8; 32];
        let request2 = ShallengeRequest {
            username: &username,
            username_len: 10,
            target_hash: &zero,
            thread_idx,
            rng_seed,
        };
        let result2 = generate_and_check_shallenge(&request2);
        assert_eq!(result2.is_better, false);
    }
}
