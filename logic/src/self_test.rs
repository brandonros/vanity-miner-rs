//! On-device / on-CPU self-test: runs known-answer tests for every logic
//! primitive against externally-validated expected values, writing
//! pass(1)/fail(0) per check into the results buffer.
//!
//! Each slot has a dedicated `check_*` function. GPU mode launches one
//! kernel per slot so an illegal-address fault localizes to a single
//! subsystem (and the kernels before it still produce reliable results
//! before the context goes sticky-errored). CPU mode's `run_self_test`
//! calls them all in sequence.

use crate::{
    BitcoinVanityKeyRequest, BitcoinVanityKeyResult, EthereumVanityKeyRequest,
    EthereumVanityKeyResult, ShallengeRequest, ShallengeResult, SolanaVanityKeyRequest,
    SolanaVanityKeyResult, base58_encode_32, compare_hashes, ed25519_derive_public_key,
    generate_and_check_bitcoin_vanity_key, generate_and_check_ethereum_vanity_key,
    generate_and_check_shallenge, generate_and_check_solana_vanity_key,
    generate_random_private_key, keccak256_64bytes, private_key_to_wif,
    ripemd160_32bytes_from_bytes, secp256k1_derive_public_key,
    secp256k1_derive_public_key_uncompressed, sha256_32_from_bytes, sha256_from_bytes,
    sha512_32bytes_from_bytes,
};

pub const SELF_TEST_NUM_CHECKS: usize = 31;

/// Slot labels in order; useful for printing results.
///
/// Slots 0–3 isolate the four primitives that the solana pipeline composes
/// (xoroshiro → sha512 → ed25519 → base58). Slots 4–9 isolate the remaining
/// primitives used by the bitcoin / ethereum / shallenge / WIF pipelines
/// (secp256k1 compressed + uncompressed, keccak256, ripemd160, and the two
/// sha256 entry points). Running all primitives first means a fault inside
/// any one surfaces in its own slot before the composed pipeline kernels
/// (slots 10+) would have inlined it.
pub const SELF_TEST_LABELS: [&str; SELF_TEST_NUM_CHECKS] = [
    "xoroshiro priv",
    "sha512 of priv",
    "ed25519 derive",
    "base58 encode pub",
    "secp256k1 compressed",
    "secp256k1 uncompressed",
    "keccak256 64bytes",
    "ripemd160 32bytes",
    "sha256 32bytes",
    "sha256 variable",
    "solana priv",
    "solana pub",
    "solana encoded",
    "ethereum priv",
    "ethereum pub",
    "ethereum address",
    "bitcoin priv",
    "bitcoin pub",
    "bitcoin pkh",
    "bitcoin encoded",
    "bitcoin matches",
    "wif compressed mainnet",
    "wif uncompressed mainnet",
    "wif compressed testnet",
    "wif uncompressed testnet",
    "shallenge hash",
    "shallenge nonce_len",
    "shallenge is_better",
    "compare_hashes lt",
    "compare_hashes gt",
    "compare_hashes eq",
];

// === Solana per-primitive bisect (slots 0-3) ===
// The `solana priv` slot ran the *whole* pipeline before checking the priv
// bytes; if that kernel faulted we couldn't tell which primitive triggered
// it. These four `check_primitive_*` functions exercise each stage in
// isolation against externally-validated intermediates, so GPU mode can
// localize a fault to xoroshiro / sha512 / ed25519 / base58.

const SOLANA_PRIMITIVE_PRIV: [u8; 32] = [
    0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48,
    0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2, 0xc4,
    0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14,
    0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7, 0x95, 0x6b,
];

const SOLANA_PRIMITIVE_HASHED_PRIV: [u8; 64] = [
    0xaa, 0xe4, 0x1d, 0x15, 0x43, 0x8a, 0x30, 0xa5,
    0x0e, 0x27, 0x4b, 0x13, 0x6d, 0x5c, 0x2a, 0x7c,
    0x36, 0x6e, 0x68, 0xbf, 0xf9, 0xa0, 0xbb, 0x05,
    0x87, 0x2c, 0x35, 0x75, 0x2e, 0x9a, 0x45, 0xa4,
    0x8c, 0x25, 0x5f, 0x21, 0xb8, 0x43, 0xfc, 0xa7,
    0x21, 0x81, 0x3f, 0xc2, 0x40, 0x3e, 0x20, 0x13,
    0xe0, 0xe8, 0x1d, 0xd6, 0xd7, 0xc9, 0xd8, 0x69,
    0xac, 0xf6, 0x03, 0x1e, 0x33, 0xb6, 0x95, 0x6a,
];

const SOLANA_PRIMITIVE_PUB: [u8; 32] = [
    0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d,
    0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0, 0x28,
    0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc,
    0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30, 0x16, 0xbb,
];

pub fn check_primitive_xoroshiro() -> u32 {
    let priv_key = generate_random_private_key(3, 583437459223573146);
    (priv_key == SOLANA_PRIMITIVE_PRIV) as u32
}

pub fn check_primitive_sha512() -> u32 {
    let hashed = sha512_32bytes_from_bytes(&SOLANA_PRIMITIVE_PRIV);
    (hashed == SOLANA_PRIMITIVE_HASHED_PRIV) as u32
}

pub fn check_primitive_ed25519() -> u32 {
    let pub_key = ed25519_derive_public_key(&SOLANA_PRIMITIVE_HASHED_PRIV);
    (pub_key == SOLANA_PRIMITIVE_PUB) as u32
}

pub fn check_primitive_base58() -> u32 {
    let expected: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
    let mut out = [0u8; 64];
    let n = base58_encode_32(&SOLANA_PRIMITIVE_PUB, &mut out);
    (n == expected.len() && bytes_eq_prefix(&out, expected)) as u32
}

// === Non-solana primitive bisect (slots 4-9) ===
// Same idea as slots 0-3, but for the primitives consumed by the bitcoin /
// ethereum / shallenge / WIF pipelines. Each KAT pair is taken from the
// per-module unit tests in the corresponding `logic/src/*.rs` file, so a
// fault here means the primitive itself is broken on the device — separate
// from a fault in a composed pipeline kernel that just inlines it.

const SECP256K1_PRIMITIVE_PRIV: [u8; 32] = [
    0x15, 0x2d, 0x53, 0x72, 0x3d, 0xa4, 0x20, 0x34,
    0x78, 0x57, 0x4b, 0x15, 0x31, 0x43, 0xa7, 0xea,
    0xa9, 0x21, 0xa8, 0xd8, 0x2c, 0x62, 0x95, 0x17,
    0xd6, 0xb1, 0x89, 0x49, 0xf0, 0x11, 0x1a, 0xbb,
];

const SECP256K1_PRIMITIVE_COMPRESSED_PUB: [u8; 33] = [
    0x03, 0x91, 0x63, 0xab, 0x44, 0x9d, 0x4b, 0x90,
    0xde, 0x13, 0xce, 0x60, 0xb5, 0x04, 0xbf, 0xc2,
    0x7a, 0x4a, 0xed, 0x37, 0x8c, 0x1f, 0x83, 0x38,
    0x68, 0x61, 0x56, 0xb9, 0x14, 0x45, 0x63, 0x7c,
    0x8d,
];

const SECP256K1_PRIMITIVE_UNCOMPRESSED_PUB: [u8; 65] = [
    0x04, 0x91, 0x63, 0xab, 0x44, 0x9d, 0x4b, 0x90,
    0xde, 0x13, 0xce, 0x60, 0xb5, 0x04, 0xbf, 0xc2,
    0x7a, 0x4a, 0xed, 0x37, 0x8c, 0x1f, 0x83, 0x38,
    0x68, 0x61, 0x56, 0xb9, 0x14, 0x45, 0x63, 0x7c,
    0x8d, 0x33, 0x27, 0x2b, 0x79, 0x99, 0x4d, 0xae,
    0x54, 0xda, 0x40, 0x11, 0xcc, 0x3e, 0x34, 0x91,
    0xcc, 0xdf, 0x3b, 0xd3, 0xfd, 0x92, 0x97, 0x8a,
    0x00, 0x87, 0x37, 0x27, 0xf9, 0x9b, 0xeb, 0x43,
    0x75,
];

const KECCAK256_PRIMITIVE_INPUT: [u8; 64] = [
    0x61, 0xa3, 0x14, 0xb0, 0x18, 0x37, 0x24, 0xea,
    0x0e, 0x5f, 0x23, 0x75, 0x84, 0xcb, 0x76, 0x09,
    0x2e, 0x25, 0x3b, 0x99, 0x78, 0x3d, 0x84, 0x6a,
    0x5b, 0x10, 0xdb, 0x15, 0x51, 0x28, 0xea, 0xfd,
    0x61, 0xa3, 0x14, 0xb0, 0x18, 0x37, 0x24, 0xea,
    0x0e, 0x5f, 0x23, 0x75, 0x84, 0xcb, 0x76, 0x09,
    0x2e, 0x25, 0x3b, 0x99, 0x78, 0x3d, 0x84, 0x6a,
    0x5b, 0x10, 0xdb, 0x15, 0x51, 0x28, 0xea, 0xfd,
];

const KECCAK256_PRIMITIVE_OUTPUT: [u8; 32] = [
    0x0f, 0x43, 0x9a, 0x98, 0x30, 0x55, 0x8b, 0x9c,
    0xd6, 0x84, 0x23, 0x28, 0xdd, 0x11, 0x58, 0x54,
    0x01, 0xc3, 0x43, 0x21, 0xa5, 0x3b, 0x29, 0x42,
    0x2a, 0xac, 0xde, 0x31, 0x06, 0x43, 0xd3, 0x73,
];

// "brandonros/000000000000000000000" — 32 ASCII bytes.
const HASH_PRIMITIVE_INPUT_32: [u8; 32] = *b"brandonros/000000000000000000000";

const RIPEMD160_PRIMITIVE_OUTPUT: [u8; 20] = [
    0xce, 0xf7, 0x32, 0xce, 0xe6, 0x7e, 0xa5, 0xd8,
    0x1d, 0x08, 0x70, 0x8b, 0x22, 0xbf, 0x1f, 0xc7,
    0x91, 0x1d, 0x32, 0x09,
];

const SHA256_PRIMITIVE_OUTPUT_32: [u8; 32] = [
    0xf7, 0xa4, 0x1d, 0xae, 0x11, 0x96, 0x28, 0x2f,
    0x0a, 0x54, 0x4a, 0x8c, 0x7f, 0x1b, 0xbf, 0x61,
    0xbd, 0xa7, 0x93, 0x07, 0xdc, 0x42, 0x4c, 0x0d,
    0x9f, 0xeb, 0xd2, 0x7b, 0x08, 0xe1, 0xbf, 0x78,
];

// 33 ASCII bytes — forces the variable-length sha256 path through one
// complete 64-byte block plus padding.
const HASH_PRIMITIVE_INPUT_33: [u8; 33] = *b"brandonros/0000000000000000000000";

const SHA256_PRIMITIVE_OUTPUT_VARIABLE: [u8; 32] = [
    0x06, 0x23, 0x89, 0x93, 0x6c, 0x51, 0x9e, 0xd7,
    0x3f, 0x33, 0x71, 0xef, 0x2e, 0x66, 0xd4, 0x38,
    0xe1, 0xcf, 0x0a, 0x66, 0x03, 0xf8, 0xb6, 0x7c,
    0x74, 0x8a, 0x5d, 0x21, 0x1e, 0x48, 0xb2, 0x9d,
];

pub fn check_primitive_secp256k1_compressed() -> u32 {
    let pub_key = secp256k1_derive_public_key(&SECP256K1_PRIMITIVE_PRIV);
    (pub_key == SECP256K1_PRIMITIVE_COMPRESSED_PUB) as u32
}

pub fn check_primitive_secp256k1_uncompressed() -> u32 {
    let pub_key = secp256k1_derive_public_key_uncompressed(&SECP256K1_PRIMITIVE_PRIV);
    (pub_key == SECP256K1_PRIMITIVE_UNCOMPRESSED_PUB) as u32
}

pub fn check_primitive_keccak256() -> u32 {
    let hash = keccak256_64bytes(&KECCAK256_PRIMITIVE_INPUT);
    (hash == KECCAK256_PRIMITIVE_OUTPUT) as u32
}

pub fn check_primitive_ripemd160() -> u32 {
    let hash = ripemd160_32bytes_from_bytes(&HASH_PRIMITIVE_INPUT_32);
    (hash == RIPEMD160_PRIMITIVE_OUTPUT) as u32
}

pub fn check_primitive_sha256_32() -> u32 {
    let hash = sha256_32_from_bytes(&HASH_PRIMITIVE_INPUT_32);
    (hash == SHA256_PRIMITIVE_OUTPUT_32) as u32
}

pub fn check_primitive_sha256_variable() -> u32 {
    let hash = sha256_from_bytes(&HASH_PRIMITIVE_INPUT_33);
    (hash == SHA256_PRIMITIVE_OUTPUT_VARIABLE) as u32
}

/// Compare the first `expected.len()` bytes of `actual` to `expected`.
fn bytes_eq_prefix(actual: &[u8; 64], expected: &[u8]) -> bool {
    let n = expected.len();
    let mut i = 0;
    while i < n {
        if actual[i] != expected[i] {
            return false;
        }
        i += 1;
    }
    true
}

// === Solana (rng_seed=583437459223573146, thread_idx=3) ===

fn solana_test() -> SolanaVanityKeyResult {
    let req = SolanaVanityKeyRequest {
        prefix: b"",
        suffix: b"",
        thread_idx: 3,
        rng_seed: 583437459223573146,
    };
    generate_and_check_solana_vanity_key(&req)
}

pub fn check_solana_priv() -> u32 {
    let expected: [u8; 32] = [
        0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48,
        0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2, 0xc4,
        0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14,
        0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7, 0x95, 0x6b,
    ];
    (solana_test().private_key == expected) as u32
}

pub fn check_solana_pub() -> u32 {
    let expected: [u8; 32] = [
        0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d,
        0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0, 0x28,
        0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc,
        0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30, 0x16, 0xbb,
    ];
    (solana_test().public_key == expected) as u32
}

pub fn check_solana_encoded() -> u32 {
    let expected: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
    let sol = solana_test();
    (sol.encoded_len == expected.len()
        && bytes_eq_prefix(&sol.encoded_public_key, expected)) as u32
}

// === Ethereum (rng_seed=15455378110306975741, thread_idx=0) ===

fn ethereum_test() -> EthereumVanityKeyResult {
    let req = EthereumVanityKeyRequest {
        prefix: b"",
        suffix: b"",
        thread_idx: 0,
        rng_seed: 15455378110306975741,
    };
    generate_and_check_ethereum_vanity_key(&req)
}

pub fn check_ethereum_priv() -> u32 {
    let expected: [u8; 32] = [
        0x1c, 0xcf, 0x23, 0x85, 0x14, 0x11, 0x73, 0x04,
        0x8c, 0x0d, 0x06, 0xc1, 0x07, 0x08, 0x69, 0xa1,
        0x6b, 0xf6, 0x3b, 0x69, 0x71, 0x66, 0x33, 0xe9,
        0xbf, 0xe7, 0x9a, 0x98, 0x13, 0xc4, 0x05, 0xab,
    ];
    (ethereum_test().private_key == expected) as u32
}

pub fn check_ethereum_pub() -> u32 {
    let expected: [u8; 64] = [
        0x88, 0xf1, 0xff, 0xe7, 0x4d, 0x7c, 0x83, 0xb6,
        0xae, 0xe0, 0xc7, 0x0f, 0x42, 0x38, 0xf5, 0xaa,
        0x91, 0x7b, 0x80, 0x62, 0xc9, 0xd3, 0x78, 0xfd,
        0xf4, 0x04, 0x2c, 0xcc, 0xdc, 0xca, 0x26, 0x39,
        0x42, 0x4c, 0x5d, 0xb5, 0x21, 0x21, 0x6a, 0xb6,
        0xb7, 0x65, 0xd9, 0xf6, 0x37, 0x8e, 0xe9, 0x26,
        0x11, 0x8a, 0xbf, 0xf8, 0xaf, 0x52, 0x4e, 0x0a,
        0x5d, 0x5e, 0x82, 0x75, 0x28, 0x6d, 0xd4, 0xc9,
    ];
    (ethereum_test().public_key == expected) as u32
}

pub fn check_ethereum_address() -> u32 {
    let expected: [u8; 20] = [
        0x55, 0x55, 0x63, 0x59, 0x0c, 0x72, 0x4a, 0x58,
        0xf7, 0xbb, 0x48, 0xb6, 0xc8, 0x47, 0xaa, 0x63,
        0x1a, 0x48, 0x65, 0x1c,
    ];
    (ethereum_test().address == expected) as u32
}

// === Bitcoin (rng_seed=13278869120712471092, thread_idx=1) ===

fn bitcoin_test() -> BitcoinVanityKeyResult {
    let req = BitcoinVanityKeyRequest {
        prefix: b"bc1q",
        suffix: b"",
        thread_idx: 1,
        rng_seed: 13278869120712471092,
    };
    generate_and_check_bitcoin_vanity_key(&req)
}

pub fn check_bitcoin_priv() -> u32 {
    let expected: [u8; 32] = [
        0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3,
        0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce, 0x8a,
        0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c,
        0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67, 0xd1, 0xdc,
    ];
    (bitcoin_test().private_key == expected) as u32
}

pub fn check_bitcoin_pub() -> u32 {
    let expected: [u8; 33] = [
        0x02, 0x54, 0x38, 0x15, 0x68, 0x27, 0x6c, 0x32,
        0xfe, 0x4a, 0x16, 0x77, 0xbb, 0x97, 0xb2, 0x62,
        0x9f, 0xcf, 0x68, 0x4e, 0x3e, 0x22, 0xcb, 0x4d,
        0x95, 0xfa, 0x1c, 0x53, 0x60, 0xa0, 0xe7, 0x79,
        0xbf,
    ];
    (bitcoin_test().public_key == expected) as u32
}

pub fn check_bitcoin_pkh() -> u32 {
    let expected: [u8; 20] = [
        0x00, 0x01, 0xb5, 0x3d, 0x6d, 0x26, 0xf1, 0x8c,
        0x85, 0xbf, 0xf2, 0xac, 0x3c, 0x57, 0x1e, 0xe7,
        0xe0, 0xc8, 0x87, 0xff,
    ];
    (bitcoin_test().public_key_hash == expected) as u32
}

pub fn check_bitcoin_encoded() -> u32 {
    let expected: &[u8] = b"bc1qqqqm20tdymccepdl72krc4c7ulsv3pllzju9s4";
    let btc = bitcoin_test();
    (btc.encoded_len == expected.len()
        && bytes_eq_prefix(&btc.encoded_public_key, expected)) as u32
}

pub fn check_bitcoin_matches() -> u32 {
    bitcoin_test().matches as u32
}

// === WIF (4 flag combinations) ===
// Standalone — doesn't depend on the bitcoin search; just feeds the known
// private key into private_key_to_wif with each flag combo.

const BITCOIN_TEST_PRIV: [u8; 32] = [
    0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3,
    0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce, 0x8a,
    0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c,
    0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67, 0xd1, 0xdc,
];

pub fn check_wif_compressed_mainnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, true, false, &mut wif_buf);
    (n == 52
        && bytes_eq_prefix(&wif_buf, b"Ky34pxSf7FLh6GFgKpvJwfDFdCw6GG4vytEh3Kt3ZzZoxw3e3WaG")) as u32
}

pub fn check_wif_uncompressed_mainnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, false, false, &mut wif_buf);
    (n == 51
        && bytes_eq_prefix(&wif_buf, b"5JEA2MGL4EDcpQr6HVywMzbVgvTJWHZA4NaTk7znSbnx3ooTWrv")) as u32
}

pub fn check_wif_compressed_testnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, true, true, &mut wif_buf);
    (n == 52
        && bytes_eq_prefix(&wif_buf, b"cPQ4HsSWYK2xFhiwiEjSJyiKFSEVviAd3vPA9kLZ57DpDg5McHdr")) as u32
}

pub fn check_wif_uncompressed_testnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, false, true, &mut wif_buf);
    (n == 51
        && bytes_eq_prefix(&wif_buf, b"91znc65seTHknUMNuqsrEb9TLap1fT6MQKSQpkMHnLXzpohhjJo")) as u32
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
    generate_and_check_shallenge(&req)
}

pub fn check_shallenge_hash() -> u32 {
    let expected: [u8; 32] = [
        0xc3, 0x75, 0x0f, 0x87, 0x11, 0xbf, 0x80, 0x9f,
        0x46, 0xde, 0x1f, 0x01, 0xec, 0xeb, 0x6f, 0x4e,
        0x6f, 0xde, 0x67, 0x0a, 0xd8, 0xa3, 0xe2, 0xa6,
        0x00, 0xa0, 0xe0, 0xb7, 0x35, 0x76, 0x54, 0xc9,
    ];
    (shallenge_test().hash == expected) as u32
}

pub fn check_shallenge_nonce_len() -> u32 {
    (shallenge_test().nonce_len == 21) as u32
}

pub fn check_shallenge_is_better() -> u32 {
    shallenge_test().is_better as u32
}

// === compare_hashes (lt / gt / eq branches) ===

pub fn check_compare_hashes_lt() -> u32 {
    let zero = [0u8; 32];
    let max = [0xffu8; 32];
    (compare_hashes(&zero, &max) == -1) as u32
}

pub fn check_compare_hashes_gt() -> u32 {
    let zero = [0u8; 32];
    let max = [0xffu8; 32];
    (compare_hashes(&max, &zero) == 1) as u32
}

pub fn check_compare_hashes_eq() -> u32 {
    let zero = [0u8; 32];
    (compare_hashes(&zero, &zero) == 0) as u32
}

pub fn run_self_test(results: &mut [u32]) {
    results[0] = check_primitive_xoroshiro();
    results[1] = check_primitive_sha512();
    results[2] = check_primitive_ed25519();
    results[3] = check_primitive_base58();
    results[4] = check_primitive_secp256k1_compressed();
    results[5] = check_primitive_secp256k1_uncompressed();
    results[6] = check_primitive_keccak256();
    results[7] = check_primitive_ripemd160();
    results[8] = check_primitive_sha256_32();
    results[9] = check_primitive_sha256_variable();
    results[10] = check_solana_priv();
    results[11] = check_solana_pub();
    results[12] = check_solana_encoded();
    results[13] = check_ethereum_priv();
    results[14] = check_ethereum_pub();
    results[15] = check_ethereum_address();
    results[16] = check_bitcoin_priv();
    results[17] = check_bitcoin_pub();
    results[18] = check_bitcoin_pkh();
    results[19] = check_bitcoin_encoded();
    results[20] = check_bitcoin_matches();
    results[21] = check_wif_compressed_mainnet();
    results[22] = check_wif_uncompressed_mainnet();
    results[23] = check_wif_compressed_testnet();
    results[24] = check_wif_uncompressed_testnet();
    results[25] = check_shallenge_hash();
    results[26] = check_shallenge_nonce_len();
    results[27] = check_shallenge_is_better();
    results[28] = check_compare_hashes_lt();
    results[29] = check_compare_hashes_gt();
    results[30] = check_compare_hashes_eq();
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn all_self_test_checks_pass_on_cpu() {
        let mut results = [0u32; SELF_TEST_NUM_CHECKS];
        run_self_test(&mut results);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(r, 1, "self-test check {} ({}) failed", i, SELF_TEST_LABELS[i]);
        }
    }
}
