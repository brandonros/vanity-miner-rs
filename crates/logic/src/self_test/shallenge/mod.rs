//! shallenge self-tests: primitives, pipeline stages, sha256 padding, and candidates.
pub(super) mod candidate_probes;
pub(super) mod comparison_probes;
mod sha256_fixtures;
pub(super) mod sha256_probes;
use crate::crypto::sha256::sha256_from_bytes;
use crate::modes::shallenge::ShallengeRequest;
use crate::modes::shallenge::ShallengeResult;
use crate::modes::shallenge::compare_hashes;
use crate::modes::shallenge::generate_and_check_shallenge;
use crate::search::xoroshiro::generate_base64_nonce;

// 33 ASCII bytes — exercises variable-length SHA-256 in one padded block.
const HASH_PRIMITIVE_INPUT_33: [u8; 33] = *b"brandonros/0000000000000000000000";

const SHA256_PRIMITIVE_OUTPUT_VARIABLE: [u8; 32] = [
    0x06, 0x23, 0x89, 0x93, 0x6c, 0x51, 0x9e, 0xd7, 0x3f, 0x33, 0x71, 0xef, 0x2e, 0x66, 0xd4, 0x38,
    0xe1, 0xcf, 0x0a, 0x66, 0x03, 0xf8, 0xb6, 0x7c, 0x74, 0x8a, 0x5d, 0x21, 0x1e, 0x48, 0xb2, 0x9d,
];

const SHALLENGE_TEST_HASH: [u8; 32] = [
    0xc3, 0x75, 0x0f, 0x87, 0x11, 0xbf, 0x80, 0x9f, 0x46, 0xde, 0x1f, 0x01, 0xec, 0xeb, 0x6f, 0x4e,
    0x6f, 0xde, 0x67, 0x0a, 0xd8, 0xa3, 0xe2, 0xa6, 0x00, 0xa0, 0xe0, 0xb7, 0x35, 0x76, 0x54, 0xc9,
];

register_self_test! {
    /// sha256 variable
    fn primitive_sha256_variable() -> u32 {
        let hash = sha256_from_bytes(&core::hint::black_box(HASH_PRIMITIVE_INPUT_33));
        (hash == SHA256_PRIMITIVE_OUTPUT_VARIABLE) as u32
    }
}

// === Shallenge (rng_seed=12345, thread_idx=0, "brandonros", target=max) ===

fn shallenge_test() -> ShallengeResult {
    let user = *b"brandonros";
    let target = [0xffu8; 32];
    let req = ShallengeRequest {
        username: &user,
        username_len: 10,
        target_hash: &target,
        thread_idx: 0,
        rng_seed: 12345,
    };
    generate_and_check_shallenge(&core::hint::black_box(req))
}

register_self_test! {
    /// shallenge hash
    fn hash() -> u32 {
        let expected = SHALLENGE_TEST_HASH;
        (shallenge_test().hash == expected) as u32
    }
}

register_self_test! {
    /// shallenge nonce_len
    fn nonce_len() -> u32 {
        // This slot tests length arithmetic; hash and nonce generation have separate checks.
        let username_len = core::hint::black_box(10usize);
        (crate::modes::shallenge::shallenge_nonce_len(username_len) == 21) as u32
    }
}

register_self_test! {
    /// shallenge is_better
    fn is_better() -> u32 {
        shallenge_test().is_better as u32
    }
}

// === compare_hashes (lt / gt / eq branches) ===

register_self_test! {
    /// compare_hashes lt
    fn compare_hashes_lt() -> u32 {
        let zero = core::hint::black_box([0u8; 32]);
        let max = core::hint::black_box([0xffu8; 32]);
        (compare_hashes(&zero, &max) == -1) as u32
    }
}

register_self_test! {
    /// compare_hashes gt
    fn compare_hashes_gt() -> u32 {
        let zero = core::hint::black_box([0u8; 32]);
        let max = core::hint::black_box([0xffu8; 32]);
        (compare_hashes(&max, &zero) == 1) as u32
    }
}

register_self_test! {
    /// compare_hashes eq
    fn compare_hashes_eq() -> u32 {
        let a = core::hint::black_box([0u8; 32]);
        let b = core::hint::black_box([0u8; 32]);
        (compare_hashes(&a, &b) == 0) as u32
    }
}

// Host output of `generate_base64_nonce(0, 12345, &mut [0u8; 21])`, the same
// (thread_idx, rng_seed) as `shallenge_test`.
const XOROSHIRO_NONCE_EXPECTED: [u8; 21] = [
    0x61, 0x63, 0x65, 0x43, 0x48, 0x73, 0x71, 0x46, 0x36, 0x67, 0x31, 0x33, 0x5a, 0x65, 0x32, 0x6e,
    0x47, 0x53, 0x4a, 0x67, 0x6d,
];

register_self_test! {
    /// xoroshiro base64 nonce
    fn xoroshiro_base64_nonce() -> u32 {
        let mut nonce = [0u8; 21];
        generate_base64_nonce(core::hint::black_box(0), core::hint::black_box(12345), &mut nonce);
        (nonce == XOROSHIRO_NONCE_EXPECTED) as u32
    }
}
