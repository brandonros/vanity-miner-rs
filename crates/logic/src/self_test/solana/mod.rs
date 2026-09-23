//! solana self-tests: primitives, pipeline stages, batch seeds, and candidates.
pub(super) mod batch_seed_probes;
pub(super) mod candidate_probes;
use super::bytes_eq_prefix;
use crate::crypto::ed25519::ed25519_derive_public_key;
use crate::crypto::sha512::sha512_32bytes_from_bytes;
use crate::encoding::base58::base58_encode_32;
use crate::modes::solana::SolanaVanityKeyRequest;
use crate::modes::solana::SolanaVanityKeyResult;
use crate::modes::solana::generate_and_check_solana_vanity_key;
use crate::search::xoroshiro::generate_random_private_key;

// Pipeline stages in isolation: xoroshiro, sha512, ed25519, and base58.

const SOLANA_PRIMITIVE_PRIV: [u8; 32] = [
    0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48, 0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2, 0xc4,
    0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14, 0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7, 0x95, 0x6b,
];

const SOLANA_PRIMITIVE_HASHED_PRIV: [u8; 64] = [
    0xaa, 0xe4, 0x1d, 0x15, 0x43, 0x8a, 0x30, 0xa5, 0x0e, 0x27, 0x4b, 0x13, 0x6d, 0x5c, 0x2a, 0x7c,
    0x36, 0x6e, 0x68, 0xbf, 0xf9, 0xa0, 0xbb, 0x05, 0x87, 0x2c, 0x35, 0x75, 0x2e, 0x9a, 0x45, 0xa4,
    0x8c, 0x25, 0x5f, 0x21, 0xb8, 0x43, 0xfc, 0xa7, 0x21, 0x81, 0x3f, 0xc2, 0x40, 0x3e, 0x20, 0x13,
    0xe0, 0xe8, 0x1d, 0xd6, 0xd7, 0xc9, 0xd8, 0x69, 0xac, 0xf6, 0x03, 0x1e, 0x33, 0xb6, 0x95, 0x6a,
];

const SOLANA_PRIMITIVE_PUB: [u8; 32] = [
    0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d, 0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0, 0x28,
    0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc, 0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30, 0x16, 0xbb,
];

register_self_test! {
    /// xoroshiro priv
    fn primitive_xoroshiro() -> u32 {
        let priv_key = generate_random_private_key(core::hint::black_box(3), core::hint::black_box(583437459223573146));
        (priv_key == SOLANA_PRIMITIVE_PRIV) as u32
    }
}

register_self_test! {
    /// sha512 of priv
    fn primitive_sha512() -> u32 {
        let hashed = sha512_32bytes_from_bytes(&core::hint::black_box(SOLANA_PRIMITIVE_PRIV));
        (hashed == SOLANA_PRIMITIVE_HASHED_PRIV) as u32
    }
}

register_self_test! {
    /// ed25519 derive
    fn primitive_ed25519() -> u32 {
        let pub_key = ed25519_derive_public_key(&core::hint::black_box(SOLANA_PRIMITIVE_HASHED_PRIV));
        (pub_key == SOLANA_PRIMITIVE_PUB) as u32
    }
}

register_self_test! {
    /// base58 encode pub
    fn primitive_base58() -> u32 {
        let expected: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
        let mut out = [0u8; 64];
        let n = base58_encode_32(&core::hint::black_box(SOLANA_PRIMITIVE_PUB), &mut out);
        (n == expected.len() && bytes_eq_prefix(&out, expected)) as u32
    }
}

// === Solana (rng_seed=583437459223573146, thread_idx=3) ===

fn solana_test() -> SolanaVanityKeyResult {
    let req = SolanaVanityKeyRequest {
        prefix: b"",
        suffix: b"",
        thread_idx: 3,
        rng_seed: 583437459223573146,
    };
    generate_and_check_solana_vanity_key(&core::hint::black_box(req))
}

register_self_test! {
    /// solana priv
    fn private_key() -> u32 {
        let expected: [u8; 32] = [
            0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48, 0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2,
            0xc4, 0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14, 0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7,
            0x95, 0x6b,
        ];
        (solana_test().private_key == expected) as u32
    }
}

register_self_test! {
    /// solana pub
    fn public_key() -> u32 {
        let expected: [u8; 32] = [
            0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d, 0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0,
            0x28, 0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc, 0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30,
            0x16, 0xbb,
        ];
        (solana_test().public_key == expected) as u32
    }
}

register_self_test! {
    /// solana encoded
    fn encoded() -> u32 {
        let expected: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
        let sol = solana_test();
        (sol.encoded_len == expected.len() && bytes_eq_prefix(&sol.encoded_public_key, expected)) as u32
    }
}
