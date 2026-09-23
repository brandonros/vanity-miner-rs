//! shallenge: sha256 (fixed, variable, padding, and streaming), the nonce
//! pipeline, hash comparison, and candidates.
use crate::crypto::sha256::{Sha256, sha256_from_bytes};
use crate::modes::shallenge::{
    ShallengeRequest, ShallengeResult, candidate, compare_hashes, generate_and_check_shallenge,
    shallenge_nonce_len,
};
use crate::search::candidate_result::CandidateResult;
use crate::search::xoroshiro::{BatchSeed, generate_base64_nonce};
use core::hint::black_box;

// 33 ASCII bytes — exercises variable-length SHA-256 in one padded block.
const HASH_INPUT_33: [u8; 33] = *b"brandonros/0000000000000000000000";
const SHA256_OUTPUT_VARIABLE: [u8; 32] = [
    0x06, 0x23, 0x89, 0x93, 0x6c, 0x51, 0x9e, 0xd7, 0x3f, 0x33, 0x71, 0xef, 0x2e, 0x66, 0xd4, 0x38,
    0xe1, 0xcf, 0x0a, 0x66, 0x03, 0xf8, 0xb6, 0x7c, 0x74, 0x8a, 0x5d, 0x21, 0x1e, 0x48, 0xb2, 0x9d,
];

// The known answer: rng_seed 12345, thread_idx 0, "brandonros", target max.
const RNG_SEED: u64 = 12345;
const THREAD_IDX: usize = 0;
const USERNAME: &[u8] = b"brandonros";
const HASH: [u8; 32] = [
    0xc3, 0x75, 0x0f, 0x87, 0x11, 0xbf, 0x80, 0x9f, 0x46, 0xde, 0x1f, 0x01, 0xec, 0xeb, 0x6f, 0x4e,
    0x6f, 0xde, 0x67, 0x0a, 0xd8, 0xa3, 0xe2, 0xa6, 0x00, 0xa0, 0xe0, 0xb7, 0x35, 0x76, 0x54, 0xc9,
];
// Host output of `generate_base64_nonce(0, 12345, &mut [0u8; 21])`.
const NONCE: [u8; 21] = [
    0x61, 0x63, 0x65, 0x43, 0x48, 0x73, 0x71, 0x46, 0x36, 0x67, 0x31, 0x33, 0x5a, 0x65, 0x32, 0x6e,
    0x47, 0x53, 0x4a, 0x67, 0x6d,
];

fn shallenge_test() -> ShallengeResult {
    let target = [0xffu8; 32];
    let request = ShallengeRequest {
        username: USERNAME,
        username_len: USERNAME.len(),
        target_hash: &target,
        thread_idx: THREAD_IDX,
        rng_seed: RNG_SEED,
    };
    generate_and_check_shallenge(&black_box(request))
}

// Digests of `bytes(range(n))`, generated independently with Python hashlib.
const SHA256_RANGE_0: [u8; 32] = [
    0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
    0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55,
];
const SHA256_RANGE_55: [u8; 32] = [
    0x46, 0x3e, 0xb2, 0x8e, 0x72, 0xf8, 0x2e, 0x0a, 0x96, 0xc0, 0xa4, 0xcc, 0x53, 0x69, 0x0c, 0x57,
    0x12, 0x81, 0x13, 0x1f, 0x67, 0x2a, 0xa2, 0x29, 0xe0, 0xd4, 0x5a, 0xe5, 0x9b, 0x59, 0x8b, 0x59,
];
const SHA256_RANGE_56: [u8; 32] = [
    0xda, 0x2a, 0xe4, 0xd6, 0xb3, 0x67, 0x48, 0xf2, 0xa3, 0x18, 0xf2, 0x3e, 0x7a, 0xb1, 0xdf, 0xdf,
    0x45, 0xac, 0xdc, 0x9d, 0x04, 0x9b, 0xd8, 0x0e, 0x59, 0xde, 0x82, 0xa6, 0x08, 0x95, 0xf5, 0x62,
];
const SHA256_RANGE_63: [u8; 32] = [
    0x29, 0xaf, 0x26, 0x86, 0xfd, 0x53, 0x37, 0x4a, 0x36, 0xb0, 0x84, 0x66, 0x94, 0xcc, 0x34, 0x21,
    0x77, 0xe4, 0x28, 0xd1, 0x64, 0x75, 0x15, 0xf0, 0x78, 0x78, 0x4d, 0x69, 0xcd, 0xb9, 0xe4, 0x88,
];
const SHA256_RANGE_64: [u8; 32] = [
    0xfd, 0xea, 0xb9, 0xac, 0xf3, 0x71, 0x03, 0x62, 0xbd, 0x26, 0x58, 0xcd, 0xc9, 0xa2, 0x9e, 0x8f,
    0x9c, 0x75, 0x7f, 0xcf, 0x98, 0x11, 0x60, 0x3a, 0x8c, 0x44, 0x7c, 0xd1, 0xd9, 0x15, 0x11, 0x08,
];
const SHA256_RANGE_65: [u8; 32] = [
    0x4b, 0xfd, 0x2c, 0x8b, 0x6f, 0x1e, 0xec, 0x7a, 0x2a, 0xfe, 0xb4, 0x8b, 0x93, 0x4e, 0xe4, 0xb2,
    0x69, 0x41, 0x82, 0x02, 0x7e, 0x6d, 0x0f, 0xc0, 0x75, 0x07, 0x4f, 0x2f, 0xab, 0xb3, 0x17, 0x81,
];
const SHA256_RANGE_256: [u8; 32] = [
    0x40, 0xaf, 0xf2, 0xe9, 0xd2, 0xd8, 0x92, 0x2e, 0x47, 0xaf, 0xd4, 0x64, 0x8e, 0x69, 0x67, 0x49,
    0x71, 0x58, 0x78, 0x5f, 0xbd, 0x1d, 0xa8, 0x70, 0xe7, 0x11, 0x02, 0x66, 0xbf, 0x94, 0x48, 0x80,
];

/// sha256 of `bytes(range(N))`, one padding case per length.
fn sha256_range<const N: usize>(expected: &[u8; 32]) -> bool {
    let input = black_box(core::array::from_fn::<_, N, _>(|i| i as u8));
    sha256_from_bytes(black_box(input.as_slice())) == *expected
}

checks! {
    /// sha256 variable
    fn primitive_sha256_variable() -> bool {
        sha256_from_bytes(&black_box(HASH_INPUT_33)) == SHA256_OUTPUT_VARIABLE
    }

    /// shallenge hash
    fn hash() -> bool {
        shallenge_test().hash == HASH
    }

    /// shallenge nonce_len
    fn nonce_len() -> bool {
        // Length arithmetic only; hash and nonce generation have separate checks.
        shallenge_nonce_len(black_box(USERNAME.len())) == 21
    }

    /// shallenge is_better
    fn is_better() -> bool {
        shallenge_test().is_better
    }

    /// compare_hashes lt
    fn compare_hashes_lt() -> bool {
        let zero = black_box([0u8; 32]);
        let max = black_box([0xffu8; 32]);
        compare_hashes(&zero, &max) == -1
    }

    /// compare_hashes gt
    fn compare_hashes_gt() -> bool {
        let zero = black_box([0u8; 32]);
        let max = black_box([0xffu8; 32]);
        compare_hashes(&max, &zero) == 1
    }

    /// compare_hashes eq
    fn compare_hashes_eq() -> bool {
        let a = black_box([0u8; 32]);
        let b = black_box([0u8; 32]);
        compare_hashes(&a, &b) == 0
    }

    /// xoroshiro base64 nonce
    fn xoroshiro_base64_nonce() -> bool {
        let mut nonce = [0u8; 21];
        generate_base64_nonce(black_box(THREAD_IDX), black_box(RNG_SEED), &mut nonce);
        nonce == NONCE
    }

    /// sha256 padding length 0
    fn sha256_padding_0() -> bool {
        sha256_range::<0>(&SHA256_RANGE_0)
    }

    /// sha256 padding length 55
    fn sha256_padding_55() -> bool {
        sha256_range::<55>(&SHA256_RANGE_55)
    }

    /// sha256 padding length 56
    fn sha256_padding_56() -> bool {
        sha256_range::<56>(&SHA256_RANGE_56)
    }

    /// sha256 padding length 63
    fn sha256_padding_63() -> bool {
        sha256_range::<63>(&SHA256_RANGE_63)
    }

    /// sha256 padding length 64
    fn sha256_padding_64() -> bool {
        sha256_range::<64>(&SHA256_RANGE_64)
    }

    /// sha256 padding length 65
    fn sha256_padding_65() -> bool {
        sha256_range::<65>(&SHA256_RANGE_65)
    }

    /// sha256 split updates across block boundary
    fn sha256_streaming_boundary() -> bool {
        // The 65-byte vector with buffered, empty, and block-completing updates.
        let input = black_box(core::array::from_fn::<_, 65, _>(|i| i as u8));
        for split in [1usize, 55, 56, 63, 64] {
            let split = black_box(split);
            let mut hash = Sha256::new();
            hash.update(&input[..split]);
            hash.update(black_box(&[] as &[u8]));
            hash.update(&input[split..]);
            if hash.finalize() != SHA256_RANGE_65 {
                return false;
            }
        }
        true
    }

    /// sha256 four full input blocks and a separate padding block
    fn sha256_multiblock() -> bool {
        sha256_range::<256>(&SHA256_RANGE_256)
    }

    /// sha256 repeated short updates and full blocks share one known answer
    fn sha256_streaming_chunks() -> bool {
        let input = black_box(core::array::from_fn::<_, 256, _>(|i| i as u8));
        // Short updates must accumulate without prematurely compressing a block;
        // the larger sizes alternate between buffered and direct compression.
        for chunk_size in [1usize, 7, 63, 64, 65] {
            let mut hash = Sha256::new();
            for chunk in input.chunks(black_box(chunk_size)) {
                hash.update(chunk);
                hash.update(black_box(&[] as &[u8]));
            }
            if hash.finalize() != SHA256_RANGE_256 {
                return false;
            }
        }
        true
    }

    /// compare hashes differing only at last byte
    fn compare_hashes_last_byte() -> bool {
        let a = black_box([0x42; 32]);
        let mut b = a;
        b[31] = black_box(0x43);
        compare_hashes(&a, &b) == -1 && compare_hashes(&b, &a) == 1
    }

    /// shallenge candidate hash nonce length and padding
    fn candidate_match() -> bool {
        // Counter 32 + lane advances to the known-answer seed.
        let seed = black_box(BatchSeed {
            seed: RNG_SEED - 1,
            width: 32,
        });
        let result = candidate(&seed, black_box(32), &black_box([255; 32]), black_box(USERNAME));
        let mut expected = [0u8; 256];
        expected[..32].copy_from_slice(&HASH);
        expected[32..53].copy_from_slice(&NONCE);
        expected[96] = 21; // u32 little-endian nonce length at byte offset 96.
        result.status == CandidateResult::STATUS_MATCH && result.bytes == expected
    }

    /// shallenge candidate equal and lower targets miss
    fn candidate_miss() -> bool {
        let seed = black_box(BatchSeed {
            seed: RNG_SEED - 1,
            width: 32,
        });
        for target in [HASH, [0; 32]] {
            let result = candidate(&seed, black_box(32), &black_box(target), black_box(USERNAME));
            if result.status != CandidateResult::STATUS_MISS || result.bytes != [0; 256] {
                return false;
            }
        }
        true
    }

    /// shallenge candidate invalid username and seed
    fn candidate_invalid() -> bool {
        let seed = black_box(BatchSeed {
            seed: RNG_SEED,
            width: 32,
        });
        let target = black_box([255; 32]);
        let result = candidate(&seed, black_box(0), &target, black_box(b""));
        if result.status != CandidateResult::STATUS_ERROR || result.bytes != [0; 256] {
            return false;
        }
        let result = candidate(&seed, black_box(0), &target, &black_box([b'a'; 31]));
        if result.status != CandidateResult::STATUS_ERROR || result.bytes != [0; 256] {
            return false;
        }
        let empty = black_box(BatchSeed {
            seed: RNG_SEED,
            width: 0,
        });
        let result = candidate(&empty, black_box(0), &target, black_box(USERNAME));
        result.status == CandidateResult::STATUS_ERROR && result.bytes == [0; 256]
    }
}
