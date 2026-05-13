//! On-device / on-CPU self-test: runs known-answer tests for every logic
//! primitive against externally-validated expected values, writing
//! pass(1)/fail(0) per check into the results buffer.
//!
//! Both CPU mode and the `kernel_self_test` GPU kernel invoke
//! `run_self_test`. If the GPU results differ from the CPU results, PTX
//! codegen has diverged from CPU behavior.

use crate::{
    BitcoinVanityKeyRequest, EthereumVanityKeyRequest, ShallengeRequest,
    SolanaVanityKeyRequest, compare_hashes, generate_and_check_bitcoin_vanity_key,
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

pub fn run_self_test(results: &mut [u32]) {
    // --- Solana (rng_seed=583437459223573146, thread_idx=3) ---
    let sol_req = SolanaVanityKeyRequest {
        prefix: b"",
        suffix: b"",
        thread_idx: 3,
        rng_seed: 583437459223573146,
    };
    let sol = generate_and_check_solana_vanity_key(&sol_req);
    let sol_exp_priv: [u8; 32] = [
        0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48,
        0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3, 0xd2, 0xc4,
        0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14,
        0xb0, 0xff, 0x7c, 0x5a, 0xb5, 0xe7, 0x95, 0x6b,
    ];
    let sol_exp_pub: [u8; 32] = [
        0x08, 0x9a, 0x23, 0xff, 0xc4, 0x22, 0xf5, 0x3d,
        0x11, 0x45, 0x87, 0x01, 0x2b, 0xb2, 0xc0, 0x28,
        0x49, 0x2f, 0xab, 0xda, 0xbe, 0x12, 0x66, 0xbc,
        0x9a, 0xd6, 0x69, 0x8a, 0xc4, 0x30, 0x16, 0xbb,
    ];
    let sol_exp_enc: &[u8] = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC";
    results[0] = (sol.private_key == sol_exp_priv) as u32;
    results[1] = (sol.public_key == sol_exp_pub) as u32;
    results[2] = (sol.encoded_len == sol_exp_enc.len()
        && bytes_eq_prefix(&sol.encoded_public_key, sol_exp_enc)) as u32;

    // --- Ethereum (rng_seed=15455378110306975741, thread_idx=0) ---
    let eth_req = EthereumVanityKeyRequest {
        prefix: b"",
        suffix: b"",
        thread_idx: 0,
        rng_seed: 15455378110306975741,
    };
    let eth = generate_and_check_ethereum_vanity_key(&eth_req);
    let eth_exp_priv: [u8; 32] = [
        0x1c, 0xcf, 0x23, 0x85, 0x14, 0x11, 0x73, 0x04,
        0x8c, 0x0d, 0x06, 0xc1, 0x07, 0x08, 0x69, 0xa1,
        0x6b, 0xf6, 0x3b, 0x69, 0x71, 0x66, 0x33, 0xe9,
        0xbf, 0xe7, 0x9a, 0x98, 0x13, 0xc4, 0x05, 0xab,
    ];
    let eth_exp_pub: [u8; 64] = [
        0x88, 0xf1, 0xff, 0xe7, 0x4d, 0x7c, 0x83, 0xb6,
        0xae, 0xe0, 0xc7, 0x0f, 0x42, 0x38, 0xf5, 0xaa,
        0x91, 0x7b, 0x80, 0x62, 0xc9, 0xd3, 0x78, 0xfd,
        0xf4, 0x04, 0x2c, 0xcc, 0xdc, 0xca, 0x26, 0x39,
        0x42, 0x4c, 0x5d, 0xb5, 0x21, 0x21, 0x6a, 0xb6,
        0xb7, 0x65, 0xd9, 0xf6, 0x37, 0x8e, 0xe9, 0x26,
        0x11, 0x8a, 0xbf, 0xf8, 0xaf, 0x52, 0x4e, 0x0a,
        0x5d, 0x5e, 0x82, 0x75, 0x28, 0x6d, 0xd4, 0xc9,
    ];
    let eth_exp_addr: [u8; 20] = [
        0x55, 0x55, 0x63, 0x59, 0x0c, 0x72, 0x4a, 0x58,
        0xf7, 0xbb, 0x48, 0xb6, 0xc8, 0x47, 0xaa, 0x63,
        0x1a, 0x48, 0x65, 0x1c,
    ];
    results[3] = (eth.private_key == eth_exp_priv) as u32;
    results[4] = (eth.public_key == eth_exp_pub) as u32;
    results[5] = (eth.address == eth_exp_addr) as u32;

    // --- Bitcoin (rng_seed=13278869120712471092, thread_idx=1) ---
    let btc_req = BitcoinVanityKeyRequest {
        prefix: b"bc1q",
        suffix: b"",
        thread_idx: 1,
        rng_seed: 13278869120712471092,
    };
    let btc = generate_and_check_bitcoin_vanity_key(&btc_req);
    let btc_exp_priv: [u8; 32] = [
        0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3,
        0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce, 0x8a,
        0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c,
        0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67, 0xd1, 0xdc,
    ];
    let btc_exp_pub: [u8; 33] = [
        0x02, 0x54, 0x38, 0x15, 0x68, 0x27, 0x6c, 0x32,
        0xfe, 0x4a, 0x16, 0x77, 0xbb, 0x97, 0xb2, 0x62,
        0x9f, 0xcf, 0x68, 0x4e, 0x3e, 0x22, 0xcb, 0x4d,
        0x95, 0xfa, 0x1c, 0x53, 0x60, 0xa0, 0xe7, 0x79,
        0xbf,
    ];
    let btc_exp_pkh: [u8; 20] = [
        0x00, 0x01, 0xb5, 0x3d, 0x6d, 0x26, 0xf1, 0x8c,
        0x85, 0xbf, 0xf2, 0xac, 0x3c, 0x57, 0x1e, 0xe7,
        0xe0, 0xc8, 0x87, 0xff,
    ];
    let btc_exp_enc: &[u8] = b"bc1qqqqm20tdymccepdl72krc4c7ulsv3pllzju9s4";
    results[6] = (btc.private_key == btc_exp_priv) as u32;
    results[7] = (btc.public_key == btc_exp_pub) as u32;
    results[8] = (btc.public_key_hash == btc_exp_pkh) as u32;
    results[9] = (btc.encoded_len == btc_exp_enc.len()
        && bytes_eq_prefix(&btc.encoded_public_key, btc_exp_enc)) as u32;
    results[10] = btc.matches as u32;

    // --- WIF (4 flag combinations) ---
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&btc_exp_priv, true, false, &mut wif_buf);
    results[11] = (n == 52
        && bytes_eq_prefix(&wif_buf, b"Ky34pxSf7FLh6GFgKpvJwfDFdCw6GG4vytEh3Kt3ZzZoxw3e3WaG")) as u32;
    wif_buf = [0u8; 64];
    let n = private_key_to_wif(&btc_exp_priv, false, false, &mut wif_buf);
    results[12] = (n == 51
        && bytes_eq_prefix(&wif_buf, b"5JEA2MGL4EDcpQr6HVywMzbVgvTJWHZA4NaTk7znSbnx3ooTWrv")) as u32;
    wif_buf = [0u8; 64];
    let n = private_key_to_wif(&btc_exp_priv, true, true, &mut wif_buf);
    results[13] = (n == 52
        && bytes_eq_prefix(&wif_buf, b"cPQ4HsSWYK2xFhiwiEjSJyiKFSEVviAd3vPA9kLZ57DpDg5McHdr")) as u32;
    wif_buf = [0u8; 64];
    let n = private_key_to_wif(&btc_exp_priv, false, true, &mut wif_buf);
    results[14] = (n == 51
        && bytes_eq_prefix(&wif_buf, b"91znc65seTHknUMNuqsrEb9TLap1fT6MQKSQpkMHnLXzpohhjJo")) as u32;

    // --- Shallenge (rng_seed=12345, thread_idx=0, "brandonros", target=max) ---
    let sh_user = *b"brandonros";
    let sh_target = [0xffu8; 32];
    let sh_req = ShallengeRequest {
        username: &sh_user,
        username_len: 10,
        target_hash: &sh_target,
        thread_idx: 0,
        rng_seed: 12345,
    };
    let sh = generate_and_check_shallenge(&sh_req);
    let sh_exp_hash: [u8; 32] = [
        0xc3, 0x75, 0x0f, 0x87, 0x11, 0xbf, 0x80, 0x9f,
        0x46, 0xde, 0x1f, 0x01, 0xec, 0xeb, 0x6f, 0x4e,
        0x6f, 0xde, 0x67, 0x0a, 0xd8, 0xa3, 0xe2, 0xa6,
        0x00, 0xa0, 0xe0, 0xb7, 0x35, 0x76, 0x54, 0xc9,
    ];
    results[15] = (sh.hash == sh_exp_hash) as u32;
    results[16] = (sh.nonce_len == 21) as u32;
    results[17] = sh.is_better as u32;

    // --- compare_hashes (lt / gt / eq branches) ---
    let zero = [0u8; 32];
    let max = [0xffu8; 32];
    results[18] = (compare_hashes(&zero, &max) == -1) as u32;
    results[19] = (compare_hashes(&max, &zero) == 1) as u32;
    results[20] = (compare_hashes(&zero, &zero) == 0) as u32;
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
