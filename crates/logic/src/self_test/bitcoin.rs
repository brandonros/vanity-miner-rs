//! bitcoin: secp256k1, ripemd160, sha256, the vanity pipeline, WIF, base58,
//! bech32, byte-pattern matching, and candidates.
use crate::crypto::ripemd160::ripemd160_32bytes_from_bytes;
use crate::crypto::secp256k1::secp256k1_derive_public_key;
use crate::crypto::sha256::sha256_32_from_bytes;
use crate::encoding::base58::base58_encode;
use crate::encoding::bech32::encode_p2wpkh_address;
use crate::modes::bitcoin::{
    BitcoinVanityKeyRequest, BitcoinVanityKeyResult, candidate,
    generate_and_check_bitcoin_vanity_key, private_key_to_wif,
};
use crate::search::vanity::check_vanity_match;
use core::hint::black_box;

// Primitive known answers, shared with the unit tests in `crate::crypto`.
const SECP256K1_PRIVATE_KEY: [u8; 32] = [
    0x15, 0x2d, 0x53, 0x72, 0x3d, 0xa4, 0x20, 0x34, 0x78, 0x57, 0x4b, 0x15, 0x31, 0x43, 0xa7, 0xea,
    0xa9, 0x21, 0xa8, 0xd8, 0x2c, 0x62, 0x95, 0x17, 0xd6, 0xb1, 0x89, 0x49, 0xf0, 0x11, 0x1a, 0xbb,
];
const SECP256K1_COMPRESSED_PUBLIC_KEY: [u8; 33] = [
    0x03, 0x91, 0x63, 0xab, 0x44, 0x9d, 0x4b, 0x90, 0xde, 0x13, 0xce, 0x60, 0xb5, 0x04, 0xbf, 0xc2,
    0x7a, 0x4a, 0xed, 0x37, 0x8c, 0x1f, 0x83, 0x38, 0x68, 0x61, 0x56, 0xb9, 0x14, 0x45, 0x63, 0x7c,
    0x8d,
];
// "brandonros/000000000000000000000" — 32 ASCII bytes.
const HASH_INPUT_32: [u8; 32] = *b"brandonros/000000000000000000000";
const RIPEMD160_OUTPUT: [u8; 20] = [
    0xce, 0xf7, 0x32, 0xce, 0xe6, 0x7e, 0xa5, 0xd8, 0x1d, 0x08, 0x70, 0x8b, 0x22, 0xbf, 0x1f, 0xc7,
    0x91, 0x1d, 0x32, 0x09,
];
const SHA256_OUTPUT_32: [u8; 32] = [
    0xf7, 0xa4, 0x1d, 0xae, 0x11, 0x96, 0x28, 0x2f, 0x0a, 0x54, 0x4a, 0x8c, 0x7f, 0x1b, 0xbf, 0x61,
    0xbd, 0xa7, 0x93, 0x07, 0xdc, 0x42, 0x4c, 0x0d, 0x9f, 0xeb, 0xd2, 0x7b, 0x08, 0xe1, 0xbf, 0x78,
];

// The known answer: this xoroshiro seed and lane produce PRIVATE_KEY.
const RNG_SEED: u64 = 13278869120712471092;
const THREAD_IDX: usize = 1;
const PRIVATE_KEY: [u8; 32] = [
    0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3, 0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce, 0x8a,
    0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c, 0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67, 0xd1, 0xdc,
];
const PUBLIC_KEY: [u8; 33] = [
    0x02, 0x54, 0x38, 0x15, 0x68, 0x27, 0x6c, 0x32, 0xfe, 0x4a, 0x16, 0x77, 0xbb, 0x97, 0xb2, 0x62,
    0x9f, 0xcf, 0x68, 0x4e, 0x3e, 0x22, 0xcb, 0x4d, 0x95, 0xfa, 0x1c, 0x53, 0x60, 0xa0, 0xe7, 0x79,
    0xbf,
];
const PUBLIC_KEY_HASH: [u8; 20] = [
    0x00, 0x01, 0xb5, 0x3d, 0x6d, 0x26, 0xf1, 0x8c, 0x85, 0xbf, 0xf2, 0xac, 0x3c, 0x57, 0x1e, 0xe7,
    0xe0, 0xc8, 0x87, 0xff,
];
const ENCODED: &[u8] = b"bc1qqqqm20tdymccepdl72krc4c7ulsv3pllzju9s4";

fn bitcoin_test() -> BitcoinVanityKeyResult {
    let request = BitcoinVanityKeyRequest {
        prefix: b"bc1q",
        suffix: b"",
        thread_idx: THREAD_IDX,
        rng_seed: RNG_SEED,
    };
    generate_and_check_bitcoin_vanity_key(&black_box(request))
}

/// WIF of the known private key with these flags.
fn wif(compressed: bool, testnet: bool, expected: &[u8]) -> bool {
    let mut out = [0u8; 64];
    let n = private_key_to_wif(
        &black_box(PRIVATE_KEY),
        black_box(compressed),
        black_box(testnet),
        &mut out,
    );
    out.get(..n) == Some(expected)
}

// Bitcoin Genesis P2PKH (mainnet) — one leading 0x00 forces the
// `num_leading_zeros` pad branch to emit a single '1' before the encoded
// numeric tail.
const BASE58_LEADZERO_INPUT: [u8; 25] = [
    0x00, 0x62, 0xE9, 0x07, 0xB1, 0x5C, 0xBF, 0x27, 0xD5, 0x42, 0x53, 0x99, 0xEB, 0xF6, 0xF0, 0xFB,
    0x50, 0xEB, 0xB8, 0x8F, 0x18, 0xC2, 0x9B, 0x7D, 0x93,
];
const BASE58_LEADZERO_EXPECTED: &[u8] = b"1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa";

// p2wpkh KAT lifted from bech32::test::should_encode_p2wpkh_correctly.
const BECH32_P2WPKH_HASH: [u8; 20] = [
    0x46, 0x04, 0x7c, 0x8a, 0x3d, 0x8e, 0xdb, 0x13, 0x4c, 0x3f, 0x1a, 0x3e, 0x7d, 0x65, 0xb0, 0xfd,
    0x74, 0x21, 0xf1, 0x27,
];
const BECH32_P2WPKH_EXPECTED: &[u8] = b"bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m";

checks! {
    /// secp256k1 compressed
    fn primitive_secp256k1_compressed() -> bool {
        secp256k1_derive_public_key(&black_box(SECP256K1_PRIVATE_KEY))
            == SECP256K1_COMPRESSED_PUBLIC_KEY
    }

    /// ripemd160 32bytes
    fn primitive_ripemd160() -> bool {
        ripemd160_32bytes_from_bytes(&black_box(HASH_INPUT_32)) == RIPEMD160_OUTPUT
    }

    /// sha256 32bytes
    fn primitive_sha256_32() -> bool {
        sha256_32_from_bytes(&black_box(HASH_INPUT_32)) == SHA256_OUTPUT_32
    }

    /// bitcoin priv
    fn private_key() -> bool {
        bitcoin_test().private_key == PRIVATE_KEY
    }

    /// bitcoin pub
    fn public_key() -> bool {
        bitcoin_test().public_key == PUBLIC_KEY
    }

    /// bitcoin pkh
    fn pkh() -> bool {
        bitcoin_test().public_key_hash == PUBLIC_KEY_HASH
    }

    /// bitcoin encoded
    fn encoded() -> bool {
        let result = bitcoin_test();
        result.encoded_public_key.get(..result.encoded_len) == Some(ENCODED)
    }

    /// bitcoin matches
    fn matches() -> bool {
        bitcoin_test().matches
    }

    /// wif compressed mainnet
    fn wif_compressed_mainnet() -> bool {
        wif(true, false, b"Ky34pxSf7FLh6GFgKpvJwfDFdCw6GG4vytEh3Kt3ZzZoxw3e3WaG")
    }

    /// wif uncompressed mainnet
    fn wif_uncompressed_mainnet() -> bool {
        wif(false, false, b"5JEA2MGL4EDcpQr6HVywMzbVgvTJWHZA4NaTk7znSbnx3ooTWrv")
    }

    /// wif compressed testnet
    fn wif_compressed_testnet() -> bool {
        wif(true, true, b"cPQ4HsSWYK2xFhiwiEjSJyiKFSEVviAd3vPA9kLZ57DpDg5McHdr")
    }

    /// wif uncompressed testnet
    fn wif_uncompressed_testnet() -> bool {
        wif(false, true, b"91znc65seTHknUMNuqsrEb9TLap1fT6MQKSQpkMHnLXzpohhjJo")
    }

    /// bech32 p2wpkh
    fn bech32_p2wpkh() -> bool {
        let mut out = [0u8; 64];
        let n = encode_p2wpkh_address(&black_box(BECH32_P2WPKH_HASH), black_box(true), &mut out);
        out.get(..n) == Some(BECH32_P2WPKH_EXPECTED)
    }

    /// base58 var-len leading-zero
    fn base58_var_len_leading_zero() -> bool {
        let mut out = [0u8; 64];
        let n = base58_encode(&black_box(BASE58_LEADZERO_INPUT), &mut out);
        out.get(..n) == Some(BASE58_LEADZERO_EXPECTED)
    }

    /// byte pattern prefix suffix and late mismatches
    fn vanity_prefix_suffix() -> bool {
        let data = black_box(*b"abcdef");
        check_vanity_match(&data, black_box(b""), black_box(b"ef"))
            && check_vanity_match(&data, black_box(b"abc"), black_box(b"def"))
            && check_vanity_match(&data, black_box(b"abcd"), black_box(b"cdef"))
            && !check_vanity_match(&data, black_box(b"abd"), black_box(b"ef"))
            && !check_vanity_match(&data, black_box(b"abc"), black_box(b"deg"))
    }

    /// byte pattern empty exact and overlong inputs
    fn vanity_length_boundaries() -> bool {
        let data = black_box(*b"abc");
        check_vanity_match(&data, black_box(b"abc"), black_box(b"abc"))
            && check_vanity_match(black_box(b""), black_box(b""), black_box(b""))
            && !check_vanity_match(&data, black_box(b"abcd"), black_box(b""))
            && !check_vanity_match(&data, black_box(b""), black_box(b"abcd"))
            && !check_vanity_match(black_box(b""), black_box(b"a"), black_box(b""))
    }

    /// bitcoin candidate match payload
    fn candidate_match() -> bool {
        super::candidate::matches(candidate, RNG_SEED, THREAD_IDX, b"bc1q", b"ju9s4", &PRIVATE_KEY)
    }

    /// bitcoin candidate suffix mismatch
    fn candidate_miss() -> bool {
        super::candidate::misses(candidate, RNG_SEED, THREAD_IDX, b"bc1q", b"ju9s5")
    }

    /// bitcoin candidate invalid seed and pattern
    fn candidate_invalid() -> bool {
        super::candidate::rejects_invalid(candidate, RNG_SEED)
    }
}
