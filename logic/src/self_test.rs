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
    SolanaVanityKeyResult, compare_hashes, generate_and_check_bitcoin_vanity_key,
    generate_and_check_ethereum_vanity_key, generate_and_check_shallenge,
    generate_and_check_solana_vanity_key, private_key_to_wif,
};

pub const SELF_TEST_NUM_CHECKS: usize = 21;

/// Slot labels in order; useful for printing results.
pub const SELF_TEST_LABELS: [&str; SELF_TEST_NUM_CHECKS] = [
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
    results[0] = check_solana_priv();
    results[1] = check_solana_pub();
    results[2] = check_solana_encoded();
    results[3] = check_ethereum_priv();
    results[4] = check_ethereum_pub();
    results[5] = check_ethereum_address();
    results[6] = check_bitcoin_priv();
    results[7] = check_bitcoin_pub();
    results[8] = check_bitcoin_pkh();
    results[9] = check_bitcoin_encoded();
    results[10] = check_bitcoin_matches();
    results[11] = check_wif_compressed_mainnet();
    results[12] = check_wif_uncompressed_mainnet();
    results[13] = check_wif_compressed_testnet();
    results[14] = check_wif_uncompressed_testnet();
    results[15] = check_shallenge_hash();
    results[16] = check_shallenge_nonce_len();
    results[17] = check_shallenge_is_better();
    results[18] = check_compare_hashes_lt();
    results[19] = check_compare_hashes_gt();
    results[20] = check_compare_hashes_eq();
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
