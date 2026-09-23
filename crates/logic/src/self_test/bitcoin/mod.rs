//! bitcoin self-tests: primitives, pipeline stages, encodings, matching, and candidates.
pub(super) mod base58_probes;
pub(super) mod candidate_probes;
pub(super) mod matching_probes;
use super::bytes_eq_prefix;
use crate::crypto::ripemd160::ripemd160_32bytes_from_bytes;
use crate::crypto::secp256k1::secp256k1_derive_public_key;
use crate::crypto::sha256::sha256_32_from_bytes;
use crate::encoding::base58::base58_encode;
use crate::encoding::bech32::encode_p2wpkh_address;
use crate::modes::bitcoin::BitcoinVanityKeyRequest;
use crate::modes::bitcoin::BitcoinVanityKeyResult;
use crate::modes::bitcoin::generate_and_check_bitcoin_vanity_key;
use crate::modes::bitcoin::private_key_to_wif;

// Primitive known answers, shared with the unit tests in `crate::crypto`.

const SECP256K1_PRIMITIVE_PRIV: [u8; 32] = [
    0x15, 0x2d, 0x53, 0x72, 0x3d, 0xa4, 0x20, 0x34, 0x78, 0x57, 0x4b, 0x15, 0x31, 0x43, 0xa7, 0xea,
    0xa9, 0x21, 0xa8, 0xd8, 0x2c, 0x62, 0x95, 0x17, 0xd6, 0xb1, 0x89, 0x49, 0xf0, 0x11, 0x1a, 0xbb,
];

const SECP256K1_PRIMITIVE_COMPRESSED_PUB: [u8; 33] = [
    0x03, 0x91, 0x63, 0xab, 0x44, 0x9d, 0x4b, 0x90, 0xde, 0x13, 0xce, 0x60, 0xb5, 0x04, 0xbf, 0xc2,
    0x7a, 0x4a, 0xed, 0x37, 0x8c, 0x1f, 0x83, 0x38, 0x68, 0x61, 0x56, 0xb9, 0x14, 0x45, 0x63, 0x7c,
    0x8d,
];

// "brandonros/000000000000000000000" — 32 ASCII bytes.
const HASH_PRIMITIVE_INPUT_32: [u8; 32] = *b"brandonros/000000000000000000000";

const RIPEMD160_PRIMITIVE_OUTPUT: [u8; 20] = [
    0xce, 0xf7, 0x32, 0xce, 0xe6, 0x7e, 0xa5, 0xd8, 0x1d, 0x08, 0x70, 0x8b, 0x22, 0xbf, 0x1f, 0xc7,
    0x91, 0x1d, 0x32, 0x09,
];

const SHA256_PRIMITIVE_OUTPUT_32: [u8; 32] = [
    0xf7, 0xa4, 0x1d, 0xae, 0x11, 0x96, 0x28, 0x2f, 0x0a, 0x54, 0x4a, 0x8c, 0x7f, 0x1b, 0xbf, 0x61,
    0xbd, 0xa7, 0x93, 0x07, 0xdc, 0x42, 0x4c, 0x0d, 0x9f, 0xeb, 0xd2, 0x7b, 0x08, 0xe1, 0xbf, 0x78,
];

register_self_test! {
    /// secp256k1 compressed
    fn primitive_secp256k1_compressed() -> u32 {
        let pub_key = secp256k1_derive_public_key(&core::hint::black_box(SECP256K1_PRIMITIVE_PRIV));
        (pub_key == SECP256K1_PRIMITIVE_COMPRESSED_PUB) as u32
    }
}

register_self_test! {
    /// ripemd160 32bytes
    fn primitive_ripemd160() -> u32 {
        let hash = ripemd160_32bytes_from_bytes(&core::hint::black_box(HASH_PRIMITIVE_INPUT_32));
        (hash == RIPEMD160_PRIMITIVE_OUTPUT) as u32
    }
}

register_self_test! {
    /// sha256 32bytes
    fn primitive_sha256_32() -> u32 {
        let hash = sha256_32_from_bytes(&core::hint::black_box(HASH_PRIMITIVE_INPUT_32));
        (hash == SHA256_PRIMITIVE_OUTPUT_32) as u32
    }
}

// === Bitcoin (rng_seed=13278869120712471092, thread_idx=1) ===

fn bitcoin_test() -> BitcoinVanityKeyResult {
    let req = BitcoinVanityKeyRequest {
        prefix: b"bc1q",
        suffix: b"",
        thread_idx: 1,
        rng_seed: 13278869120712471092,
    };
    generate_and_check_bitcoin_vanity_key(&core::hint::black_box(req))
}

register_self_test! {
    /// bitcoin priv
    fn private_key() -> u32 {
        let expected: [u8; 32] = [
            0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3, 0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce,
            0x8a, 0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c, 0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67,
            0xd1, 0xdc,
        ];
        (bitcoin_test().private_key == expected) as u32
    }
}

register_self_test! {
    /// bitcoin pub
    fn public_key() -> u32 {
        let expected: [u8; 33] = [
            0x02, 0x54, 0x38, 0x15, 0x68, 0x27, 0x6c, 0x32, 0xfe, 0x4a, 0x16, 0x77, 0xbb, 0x97, 0xb2,
            0x62, 0x9f, 0xcf, 0x68, 0x4e, 0x3e, 0x22, 0xcb, 0x4d, 0x95, 0xfa, 0x1c, 0x53, 0x60, 0xa0,
            0xe7, 0x79, 0xbf,
        ];
        (bitcoin_test().public_key == expected) as u32
    }
}

register_self_test! {
    /// bitcoin pkh
    fn pkh() -> u32 {
        let expected: [u8; 20] = [
            0x00, 0x01, 0xb5, 0x3d, 0x6d, 0x26, 0xf1, 0x8c, 0x85, 0xbf, 0xf2, 0xac, 0x3c, 0x57, 0x1e,
            0xe7, 0xe0, 0xc8, 0x87, 0xff,
        ];
        (bitcoin_test().public_key_hash == expected) as u32
    }
}

register_self_test! {
    /// bitcoin encoded
    fn encoded() -> u32 {
        let expected: &[u8] = b"bc1qqqqm20tdymccepdl72krc4c7ulsv3pllzju9s4";
        let btc = bitcoin_test();
        (btc.encoded_len == expected.len() && bytes_eq_prefix(&btc.encoded_public_key, expected)) as u32
    }
}

register_self_test! {
    /// bitcoin matches
    fn matches() -> u32 {
        bitcoin_test().matches as u32
    }
}

// === WIF (4 flag combinations) ===
// Standalone — doesn't depend on the bitcoin search; just feeds the known
// private key into private_key_to_wif with each flag combo.

const BITCOIN_TEST_PRIV: [u8; 32] = [
    0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3, 0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce, 0x8a,
    0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c, 0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67, 0xd1, 0xdc,
];

register_self_test! {
    /// wif compressed mainnet
    fn wif_compressed_mainnet() -> u32 {
        let mut wif_buf = [0u8; 64];
        let n = private_key_to_wif(&BITCOIN_TEST_PRIV, true, false, &mut wif_buf);
        (n == 52
            && bytes_eq_prefix(
                &wif_buf,
                b"Ky34pxSf7FLh6GFgKpvJwfDFdCw6GG4vytEh3Kt3ZzZoxw3e3WaG",
            )) as u32
    }
}

register_self_test! {
    /// wif uncompressed mainnet
    fn wif_uncompressed_mainnet() -> u32 {
        let mut wif_buf = [0u8; 64];
        let n = private_key_to_wif(&BITCOIN_TEST_PRIV, false, false, &mut wif_buf);
        (n == 51
            && bytes_eq_prefix(
                &wif_buf,
                b"5JEA2MGL4EDcpQr6HVywMzbVgvTJWHZA4NaTk7znSbnx3ooTWrv",
            )) as u32
    }
}

register_self_test! {
    /// wif compressed testnet
    fn wif_compressed_testnet() -> u32 {
        let mut wif_buf = [0u8; 64];
        let n = private_key_to_wif(&BITCOIN_TEST_PRIV, true, true, &mut wif_buf);
        (n == 52
            && bytes_eq_prefix(
                &wif_buf,
                b"cPQ4HsSWYK2xFhiwiEjSJyiKFSEVviAd3vPA9kLZ57DpDg5McHdr",
            )) as u32
    }
}

register_self_test! {
    /// wif uncompressed testnet
    fn wif_uncompressed_testnet() -> u32 {
        let mut wif_buf = [0u8; 64];
        let n = private_key_to_wif(&BITCOIN_TEST_PRIV, false, true, &mut wif_buf);
        (n == 51
            && bytes_eq_prefix(
                &wif_buf,
                b"91znc65seTHknUMNuqsrEb9TLap1fT6MQKSQpkMHnLXzpohhjJo",
            )) as u32
    }
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

register_self_test! {
    /// bech32 p2wpkh
    fn bech32_p2wpkh() -> u32 {
        let mut out = [0u8; 64];
        let n = encode_p2wpkh_address(&BECH32_P2WPKH_HASH, true, &mut out);
        if n != BECH32_P2WPKH_EXPECTED.len() {
            return 0;
        }
        let mut i = 0;
        while i < n {
            if out[i] != BECH32_P2WPKH_EXPECTED[i] {
                return 0;
            }
            i += 1;
        }
        1
    }
}
