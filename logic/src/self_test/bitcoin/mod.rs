//! bitcoin self-tests: primitives, pipeline stages, and regressions.
use super::IdxProbe;
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

// === Non-solana primitive bisect (slots 4-9) ===
// Same idea as slots 0-3, but for the primitives consumed by the bitcoin /
// ethereum / shallenge / WIF pipelines. Each KAT pair is taken from the
// per-module unit tests in the corresponding `logic/src/*.rs` file, so a
// fault here means the primitive itself is broken on the device — separate
// from a fault in a composed pipeline kernel that just inlines it.

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

pub fn check_primitive_secp256k1_compressed() -> u32 {
    let pub_key = secp256k1_derive_public_key(&SECP256K1_PRIMITIVE_PRIV);
    (pub_key == SECP256K1_PRIMITIVE_COMPRESSED_PUB) as u32
}

pub fn check_primitive_ripemd160() -> u32 {
    let hash = ripemd160_32bytes_from_bytes(&core::hint::black_box(HASH_PRIMITIVE_INPUT_32));
    (hash == RIPEMD160_PRIMITIVE_OUTPUT) as u32
}

pub fn check_primitive_sha256_32() -> u32 {
    let hash = sha256_32_from_bytes(&core::hint::black_box(HASH_PRIMITIVE_INPUT_32));
    (hash == SHA256_PRIMITIVE_OUTPUT_32) as u32
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
        0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3, 0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce,
        0x8a, 0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c, 0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67,
        0xd1, 0xdc,
    ];
    (bitcoin_test().private_key == expected) as u32
}

pub fn check_bitcoin_pub() -> u32 {
    let expected: [u8; 33] = [
        0x02, 0x54, 0x38, 0x15, 0x68, 0x27, 0x6c, 0x32, 0xfe, 0x4a, 0x16, 0x77, 0xbb, 0x97, 0xb2,
        0x62, 0x9f, 0xcf, 0x68, 0x4e, 0x3e, 0x22, 0xcb, 0x4d, 0x95, 0xfa, 0x1c, 0x53, 0x60, 0xa0,
        0xe7, 0x79, 0xbf,
    ];
    (bitcoin_test().public_key == expected) as u32
}

pub fn check_bitcoin_pkh() -> u32 {
    let expected: [u8; 20] = [
        0x00, 0x01, 0xb5, 0x3d, 0x6d, 0x26, 0xf1, 0x8c, 0x85, 0xbf, 0xf2, 0xac, 0x3c, 0x57, 0x1e,
        0xe7, 0xe0, 0xc8, 0x87, 0xff,
    ];
    (bitcoin_test().public_key_hash == expected) as u32
}

pub fn check_bitcoin_encoded() -> u32 {
    let expected: &[u8] = b"bc1qqqqm20tdymccepdl72krc4c7ulsv3pllzju9s4";
    let btc = bitcoin_test();
    (btc.encoded_len == expected.len() && bytes_eq_prefix(&btc.encoded_public_key, expected)) as u32
}

pub fn check_bitcoin_matches() -> u32 {
    bitcoin_test().matches as u32
}

// === WIF (4 flag combinations) ===
// Standalone — doesn't depend on the bitcoin search; just feeds the known
// private key into private_key_to_wif with each flag combo.

const BITCOIN_TEST_PRIV: [u8; 32] = [
    0x36, 0x32, 0xf6, 0x6f, 0xed, 0x3b, 0x77, 0xf3, 0x30, 0x9c, 0x86, 0xd7, 0x08, 0xfc, 0xce, 0x8a,
    0x07, 0x1a, 0x61, 0xa1, 0xa9, 0x4a, 0xdd, 0x0c, 0xb4, 0x5f, 0x95, 0x7c, 0x34, 0x67, 0xd1, 0xdc,
];

pub fn check_wif_compressed_mainnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, true, false, &mut wif_buf);
    (n == 52
        && bytes_eq_prefix(
            &wif_buf,
            b"Ky34pxSf7FLh6GFgKpvJwfDFdCw6GG4vytEh3Kt3ZzZoxw3e3WaG",
        )) as u32
}

pub fn check_wif_uncompressed_mainnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, false, false, &mut wif_buf);
    (n == 51
        && bytes_eq_prefix(
            &wif_buf,
            b"5JEA2MGL4EDcpQr6HVywMzbVgvTJWHZA4NaTk7znSbnx3ooTWrv",
        )) as u32
}

pub fn check_wif_compressed_testnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, true, true, &mut wif_buf);
    (n == 52
        && bytes_eq_prefix(
            &wif_buf,
            b"cPQ4HsSWYK2xFhiwiEjSJyiKFSEVviAd3vPA9kLZ57DpDg5McHdr",
        )) as u32
}

pub fn check_wif_uncompressed_testnet() -> u32 {
    let mut wif_buf = [0u8; 64];
    let n = private_key_to_wif(&BITCOIN_TEST_PRIV, false, true, &mut wif_buf);
    (n == 51
        && bytes_eq_prefix(
            &wif_buf,
            b"91znc65seTHknUMNuqsrEb9TLap1fT6MQKSQpkMHnLXzpohhjJo",
        )) as u32
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

pub fn check_bech32_p2wpkh() -> u32 {
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

// Slot 74: full k256 derive for scalar=1. Compressed public key must
// equal the well-known secp256k1 generator G.
const SECP256K1_GENERATOR_COMPRESSED: [u8; 33] = [
    0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
    0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17,
    0x98,
];

// Slot 75: full k256 derive for scalar=2. Compressed public key must
// equal 2G (one more doubling beyond slot 74). A 74-PASS / 75-FAIL split
// pinpoints the doubling formula; a 74-FAIL / 75-FAIL means scalar mult
// is broken even for the trivial-scalar case.
const SECP256K1_TWO_G_COMPRESSED: [u8; 33] = [
    0x02, 0xC6, 0x04, 0x7F, 0x94, 0x41, 0xED, 0x7D, 0x6D, 0x30, 0x45, 0x40, 0x6E, 0x95, 0xC0, 0x7C,
    0xD8, 0x5C, 0x77, 0x8E, 0x4B, 0x8C, 0xEF, 0x3C, 0xA7, 0xAB, 0xAC, 0x09, 0xB9, 0x5C, 0x70, 0x9E,
    0xE5,
];

// Slot 76: bare `&'static [u64; 5]` runtime-indexed read. The simplest
// possible test of the "element-width > 1 byte breaks &'static reads"
// hypothesis. No struct wrapper, no arithmetic on the result.
static STATIC_U64_TABLE: [u64; 5] = [
    0x0123_4567_89AB_CDEF,
    0xFEDC_BA98_7654_3210,
    0x1111_2222_3333_4444,
    0xAAAA_BBBB_CCCC_DDDD,
    0xDEAD_BEEF_CAFE_BABE,
];

// Slot 77: same but wrapped in a single-field tuple struct — matches
// dalek's `Scalar52(pub(crate) [u64; 5])` newtype shape. If 76 PASSes
// and 77 FAILs, the bug is specifically in field projection through a
// newtype, not in the underlying array.
#[repr(transparent)]
pub struct U64Wrap5(pub [u64; 5]);

static STATIC_U64_WRAPPED: U64Wrap5 = U64Wrap5([
    0x0123_4567_89AB_CDEF,
    0xFEDC_BA98_7654_3210,
    0x1111_2222_3333_4444,
    0xAAAA_BBBB_CCCC_DDDD,
    0xDEAD_BEEF_CAFE_BABE,
]);

// Slot 91: focused Index/IndexMut trait dispatch probe on a tuple
// struct. Mirrors dalek's Scalar52 Index impl shape EXACTLY: tuple
// struct wrapping `[u64; 5]`, Index returns `&u64`, IndexMut returns
// `&mut u64`. If this FAILs, trait dispatch on `[i]` syntax is broken
// on the cuda-oxide alpha-NVPTX backend — explains why dalek (uses
// `a[i]`) fails while our port (uses `a.0[i]`) passes.

// Slot 96: `EncodedPoint::from_affine_coordinates(&GX_bytes, &GY_bytes,
// compress=true)` with hardcoded generator-x/y. Bypasses AffinePoint's
// own `to_encoded_point` (which goes through `is_identity`+
// `conditional_select`) and tests just the EncodedPoint construction.
//
// If 96 PASSes and 93 FAILs, the bug is in `is_identity`/`conditional_
// select` (slot 95 should then also FAIL). If 96 FAILs, EncodedPoint
// construction itself is broken.
const SECP256K1_GX_BYTES: [u8; 32] = [
    0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
    0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98,
];

const SECP256K1_GY_BYTES: [u8; 32] = [
    0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65, 0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08, 0xA8,
    0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19, 0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10, 0xD4, 0xB8,
];

// Slot 101: probes the exact `y.as_slice().last()` shape inside
// `Tag::compress_y`. Pass a `&GenericArray<u8, U32>` to a function, do
// `as_slice().last()` inside. Slot 99 tested write-side copy; this
// tests read-side slice access via Deref then `.last()`.
#[inline(never)]
fn last_via_as_slice(
    ga: &k256::elliptic_curve::generic_array::GenericArray<
        u8,
        k256::elliptic_curve::generic_array::typenum::U32,
    >,
) -> u8 {
    *ga.as_slice().last().expect("non-empty")
}

/// Write only this mode's stable result slots.
pub fn run(results: &mut [u32]) {
    results[4] = check_primitive_secp256k1_compressed();
    results[7] = check_primitive_ripemd160();
    results[8] = check_primitive_sha256_32();
    results[16] = check_bitcoin_priv();
    results[17] = check_bitcoin_pub();
    results[18] = check_bitcoin_pkh();
    results[19] = check_bitcoin_encoded();
    results[20] = check_bitcoin_matches();
    results[21] = check_wif_compressed_mainnet();
    results[22] = check_wif_uncompressed_mainnet();
    results[23] = check_wif_compressed_testnet();
    results[24] = check_wif_uncompressed_testnet();
    results[42] = check_base58_var_len_leading_zero();
    results[45] = check_bech32_p2wpkh();
    results[73] = check_k256_secret_from_bytes_one();
    results[74] = check_k256_derive_scalar_one();
    results[75] = check_k256_derive_scalar_two();
    results[76] = check_static_u64_array_lookup();
    results[77] = check_static_struct_wrapped_u64_lookup();
    results[78] = check_k256_encode_generator();
    results[79] = check_k256_double_generator();
    results[80] = check_k256_scalar_one_round_trip();
    results[93] = check_k256_affine_generator_encode();
    results[94] = check_subtle_choice_u8_into_bool();
    results[95] = check_subtle_conditional_select_u64();
    results[96] = check_k256_encoded_point_from_affine_coords();
    results[97] = check_index_trait_const_indices();
    results[98] = check_generic_array_basic_index();
    results[99] = check_generic_array_copy_from_slice();
    results[100] = check_from_affine_coords_replica();
    results[101] = check_generic_array_as_slice_last();
    results[104] = check_field_bytes_into_conversion();
    results[110] = check_generic_array_copy_from_ga_source();
}

mod layout_probes;
use layout_probes::*;

mod secp256k1_probes;
use secp256k1_probes::*;

mod base58_probes;
use base58_probes::*;
