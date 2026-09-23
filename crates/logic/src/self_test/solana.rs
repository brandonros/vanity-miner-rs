//! solana: xoroshiro, sha512, ed25519, base58, the vanity pipeline, batch
//! seeds, and candidates.
use crate::crypto::ed25519::ed25519_derive_public_key;
use crate::crypto::sha512::sha512_32bytes_from_bytes;
use crate::encoding::base58::base58_encode_32;
use crate::modes::solana::{
    SolanaVanityKeyRequest, SolanaVanityKeyResult, candidate, generate_and_check_solana_vanity_key,
};
use crate::search::xoroshiro::{BatchSeed, generate_random_private_key};
use core::hint::black_box;

// The known answer: this xoroshiro seed and lane produce PRIVATE_KEY.
const RNG_SEED: u64 = 583437459223573146;
const THREAD_IDX: usize = 3;

const PRIVATE_KEY: [u8; 32] = [
    0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48, 0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2, 0xc4,
    0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14, 0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7, 0x95, 0x6b,
];
const HASHED_PRIVATE_KEY: [u8; 64] = [
    0xaa, 0xe4, 0x1d, 0x15, 0x43, 0x8a, 0x30, 0xa5, 0x0e, 0x27, 0x4b, 0x13, 0x6d, 0x5c, 0x2a, 0x7c,
    0x36, 0x6e, 0x68, 0xbf, 0xf9, 0xa0, 0xbb, 0x05, 0x87, 0x2c, 0x35, 0x75, 0x2e, 0x9a, 0x45, 0xa4,
    0x8c, 0x25, 0x5f, 0x21, 0xb8, 0x43, 0xfc, 0xa7, 0x21, 0x81, 0x3f, 0xc2, 0x40, 0x3e, 0x20, 0x13,
    0xe0, 0xe8, 0x1d, 0xd6, 0xd7, 0xc9, 0xd8, 0x69, 0xac, 0xf6, 0x03, 0x1e, 0x33, 0xb6, 0x95, 0x6a,
];
const PUBLIC_KEY: [u8; 32] = [
    0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d, 0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0, 0x28,
    0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc, 0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30, 0x16, 0xbb,
];
const ENCODED: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";

fn solana_test() -> SolanaVanityKeyResult {
    let request = SolanaVanityKeyRequest {
        prefix: b"",
        suffix: b"",
        thread_idx: THREAD_IDX,
        rng_seed: RNG_SEED,
    };
    generate_and_check_solana_vanity_key(&black_box(request))
}

checks! {
    /// xoroshiro priv
    fn primitive_xoroshiro() -> bool {
        generate_random_private_key(black_box(THREAD_IDX), black_box(RNG_SEED)) == PRIVATE_KEY
    }

    /// sha512 of priv
    fn primitive_sha512() -> bool {
        sha512_32bytes_from_bytes(&black_box(PRIVATE_KEY)) == HASHED_PRIVATE_KEY
    }

    /// ed25519 derive
    fn primitive_ed25519() -> bool {
        ed25519_derive_public_key(&black_box(HASHED_PRIVATE_KEY)) == PUBLIC_KEY
    }

    /// base58 encode pub
    fn primitive_base58() -> bool {
        let mut out = [0u8; 64];
        let n = base58_encode_32(&black_box(PUBLIC_KEY), &mut out);
        out.get(..n) == Some(ENCODED)
    }

    /// solana priv
    fn private_key() -> bool {
        solana_test().private_key == PRIVATE_KEY
    }

    /// solana pub
    fn public_key() -> bool {
        solana_test().public_key == PUBLIC_KEY
    }

    /// solana encoded
    fn encoded() -> bool {
        let result = solana_test();
        result.encoded_public_key.get(..result.encoded_len) == Some(ENCODED)
    }

    /// batch seed lane boundary and seed wrap
    fn batch_seed_boundary() -> bool {
        let seed = black_box(BatchSeed {
            seed: u64::MAX,
            width: 32,
        });
        seed.position(black_box(31)) == Some((u64::MAX, 31))
            && seed.position(black_box(32)) == Some((0, 0))
            && seed.position(black_box(33)) == Some((0, 1))
            && seed.position(black_box(65)) == Some((1, 1))
    }

    /// batch seed invalid and maximum widths
    fn batch_seed_invalid_width() -> bool {
        let zero = black_box(BatchSeed { seed: 7, width: 0 });
        let large = black_box(BatchSeed {
            seed: 7,
            width: u32::MAX as u64 + 1,
        });
        let max = black_box(BatchSeed {
            seed: 7,
            width: u32::MAX as u64,
        });
        zero.position(black_box(0)).is_none()
            && large.position(black_box(0)).is_none()
            && max.position(black_box(u32::MAX as u64)) == Some((8, 0))
    }

    /// solana candidate match payload
    fn candidate_match() -> bool {
        super::candidate::matches(candidate, RNG_SEED, THREAD_IDX, b"aaa", b"PjNFC", &PRIVATE_KEY)
    }

    /// solana candidate suffix mismatch
    fn candidate_miss() -> bool {
        super::candidate::misses(candidate, RNG_SEED, THREAD_IDX, b"aaa", b"PjNFD")
    }

    /// solana candidate invalid seed and pattern
    fn candidate_invalid() -> bool {
        super::candidate::rejects_invalid(candidate, RNG_SEED)
    }
}
