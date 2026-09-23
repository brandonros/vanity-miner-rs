//! rsa pss: sha256, MGF1, PSS encoding, salt counters, CRT signing with
//! fault detection, and the candidate pipeline.
use super::{SAMPLE_SHA256, record_candidate};
use crate::crypto::rsa_crt::Rsa2048Crt;
use crate::crypto::rsa_pss::{PssError, encode_sha256, mgf1_sha256};
use crate::crypto::sha256::Sha256;
use crate::modes::rsa_pss::{RsaPssRequest, rsa_pss};
use crate::search::hex_pattern::HexPattern;
use crate::search::message_window::WindowError;
use crate::search::salt_counter::write_salt_counter;
use core::hint::black_box;

// A 2048-bit CRT key.
const RSA_P: [u8; 128] = [
    230, 193, 25, 229, 124, 107, 185, 65, 204, 31, 198, 54, 106, 167, 39, 233, 84, 132, 101, 34,
    89, 136, 238, 48, 49, 10, 43, 143, 250, 120, 19, 84, 197, 231, 31, 217, 38, 224, 95, 129, 250,
    28, 133, 194, 217, 115, 80, 51, 39, 160, 146, 58, 120, 28, 175, 190, 229, 38, 189, 202, 63, 77,
    104, 240, 79, 217, 163, 217, 100, 176, 242, 149, 210, 34, 105, 187, 157, 12, 3, 23, 65, 99, 54,
    236, 214, 48, 32, 6, 140, 205, 10, 155, 142, 10, 176, 39, 38, 52, 244, 177, 102, 71, 183, 96,
    68, 4, 119, 37, 52, 116, 121, 125, 74, 61, 228, 6, 244, 130, 181, 166, 46, 12, 64, 58, 130, 44,
    60, 129,
];
const RSA_Q: [u8; 128] = [
    253, 245, 78, 185, 87, 127, 254, 215, 86, 5, 89, 212, 109, 218, 23, 57, 205, 114, 61, 94, 117,
    154, 248, 112, 189, 241, 159, 245, 98, 17, 198, 201, 98, 167, 7, 49, 168, 106, 233, 124, 210,
    248, 53, 13, 187, 175, 230, 178, 74, 156, 28, 92, 237, 22, 193, 14, 4, 211, 21, 33, 142, 107,
    132, 218, 168, 36, 206, 27, 8, 41, 146, 190, 48, 250, 160, 192, 237, 225, 57, 218, 234, 13,
    221, 46, 146, 121, 147, 254, 227, 22, 64, 90, 113, 179, 192, 197, 168, 210, 122, 232, 239, 140,
    89, 109, 95, 254, 125, 110, 175, 43, 92, 161, 55, 131, 189, 11, 107, 248, 237, 72, 169, 24,
    214, 37, 163, 33, 4, 3,
];
const RSA_DP: [u8; 128] = [
    30, 179, 12, 13, 242, 234, 165, 255, 241, 247, 60, 56, 155, 33, 215, 246, 123, 239, 13, 65,
    111, 165, 255, 2, 92, 83, 221, 13, 207, 165, 207, 244, 148, 75, 182, 121, 254, 105, 21, 107,
    198, 105, 171, 6, 12, 70, 2, 34, 124, 32, 41, 180, 43, 172, 173, 88, 249, 69, 153, 52, 98, 13,
    155, 107, 117, 68, 32, 63, 88, 221, 13, 185, 14, 246, 211, 24, 73, 193, 130, 91, 194, 176, 63,
    107, 108, 231, 132, 43, 58, 234, 64, 130, 7, 120, 182, 140, 206, 209, 118, 200, 111, 50, 21,
    92, 114, 80, 33, 197, 202, 95, 192, 150, 1, 157, 144, 75, 184, 149, 247, 50, 191, 167, 147,
    151, 245, 31, 139, 129,
];
const RSA_DQ: [u8; 128] = [
    202, 45, 22, 165, 58, 225, 173, 50, 37, 75, 81, 62, 106, 205, 235, 27, 155, 81, 77, 69, 251,
    254, 137, 182, 199, 160, 170, 165, 161, 147, 97, 92, 69, 235, 143, 48, 155, 81, 177, 220, 43,
    224, 105, 236, 42, 245, 88, 133, 172, 28, 40, 0, 90, 199, 120, 157, 254, 125, 69, 31, 87, 208,
    133, 13, 198, 58, 182, 210, 146, 102, 246, 105, 239, 120, 55, 174, 32, 50, 227, 234, 65, 215,
    19, 180, 213, 235, 112, 68, 196, 232, 156, 6, 228, 234, 35, 151, 184, 85, 8, 128, 148, 228,
    120, 245, 56, 44, 212, 18, 223, 229, 119, 114, 233, 77, 57, 156, 244, 50, 82, 69, 60, 73, 135,
    130, 163, 33, 207, 157,
];
const RSA_Q_INV: [u8; 128] = [
    189, 174, 2, 5, 67, 155, 74, 34, 132, 73, 107, 38, 38, 65, 28, 161, 194, 99, 97, 245, 75, 245,
    48, 117, 137, 195, 150, 39, 211, 40, 45, 25, 93, 63, 246, 193, 246, 114, 147, 43, 104, 81, 38,
    235, 11, 138, 31, 85, 113, 121, 19, 87, 166, 169, 72, 0, 117, 104, 83, 168, 68, 220, 159, 99,
    23, 126, 17, 143, 107, 184, 252, 112, 235, 207, 38, 185, 120, 63, 31, 172, 178, 229, 133, 70,
    250, 81, 179, 142, 80, 7, 234, 208, 102, 170, 34, 206, 69, 193, 46, 68, 144, 181, 219, 236, 78,
    203, 203, 118, 107, 202, 141, 246, 80, 198, 139, 136, 187, 175, 202, 11, 35, 52, 71, 47, 168,
    143, 168, 118,
];
const MGF_PARTIAL: [u8; 50] = [
    0x6d, 0x31, 0xe2, 0xb2, 0xfe, 0xac, 0x18, 0x8d, 0x86, 0x8a, 0x4e, 0x91, 0x50, 0xde, 0x28, 0xbc,
    0xbd, 0xa2, 0x03, 0xa7, 0x29, 0xc0, 0x2c, 0x06, 0x48, 0x3a, 0x03, 0x2f, 0x9d, 0x09, 0x37, 0x95,
    0xd1, 0xc5, 0xd1, 0xc7, 0xcb, 0x04, 0x2c, 0x91, 0x26, 0x78, 0x9c, 0x2c, 0x55, 0x4d, 0xad, 0x43,
    0x57, 0xd9,
];
const PSS_SALT32: [u8; 256] = [
    0x4f, 0x61, 0xa7, 0x06, 0xd0, 0xe2, 0x88, 0xb2, 0xda, 0x2a, 0x30, 0x74, 0x04, 0xbf, 0x77, 0x30,
    0x4e, 0x28, 0xec, 0xd4, 0x45, 0xad, 0x93, 0xf4, 0xed, 0x34, 0x51, 0xac, 0xee, 0x24, 0x73, 0x61,
    0x78, 0x28, 0xb1, 0x00, 0x93, 0x9c, 0xee, 0x41, 0x84, 0xf0, 0x05, 0x2e, 0x70, 0xcd, 0xcc, 0xfb,
    0xec, 0xc7, 0x4b, 0x28, 0xdc, 0x4a, 0xac, 0x81, 0x8d, 0x8c, 0x8b, 0xdc, 0xf1, 0x41, 0xd2, 0x20,
    0x90, 0x14, 0x4e, 0x09, 0x76, 0x62, 0x85, 0x6a, 0xbf, 0xc3, 0x97, 0x7c, 0x43, 0x6c, 0x4a, 0x83,
    0x19, 0x9b, 0xbe, 0xeb, 0xd9, 0xe6, 0xcb, 0xad, 0x88, 0x60, 0xd2, 0x50, 0x6c, 0xb7, 0x2c, 0xec,
    0xd7, 0xca, 0xb4, 0x14, 0xc8, 0xf9, 0xcc, 0xd9, 0xc1, 0xb2, 0x95, 0xef, 0x4e, 0x5e, 0xea, 0x57,
    0xca, 0x8d, 0x42, 0x13, 0x10, 0xb2, 0x31, 0xad, 0x6a, 0xb7, 0x5e, 0x8c, 0xdd, 0xa5, 0x18, 0x8f,
    0x4e, 0xbf, 0x83, 0x40, 0xd6, 0xd5, 0x64, 0x5d, 0xa2, 0x15, 0x87, 0x87, 0x85, 0x34, 0x8a, 0x9d,
    0x9c, 0x31, 0x4e, 0x4a, 0x45, 0x0a, 0x65, 0x2c, 0xe7, 0x18, 0xef, 0xfc, 0x9c, 0x63, 0x94, 0x72,
    0x77, 0xda, 0xf3, 0x2f, 0x4d, 0x6b, 0x7d, 0x55, 0x53, 0xbd, 0xd9, 0xe5, 0x1a, 0x83, 0xb6, 0xb9,
    0xc6, 0xd1, 0x46, 0x45, 0x91, 0x78, 0xa5, 0xf6, 0xef, 0xa6, 0x0b, 0xc2, 0xfa, 0x69, 0x02, 0x54,
    0x9d, 0x52, 0x1e, 0xdb, 0x24, 0x4c, 0xfe, 0x86, 0xb6, 0x1c, 0x07, 0x94, 0x86, 0x65, 0x35, 0xcb,
    0x1c, 0xea, 0xbd, 0x0a, 0x41, 0xc1, 0x4f, 0x87, 0xe0, 0x80, 0xee, 0x34, 0x95, 0x12, 0xa4, 0x37,
    0x4e, 0x09, 0x1c, 0xeb, 0xda, 0x7a, 0xa4, 0x58, 0xb4, 0xe6, 0xc6, 0xac, 0x93, 0x51, 0x69, 0x15,
    0xd7, 0xaf, 0x61, 0xa3, 0x79, 0xc5, 0xea, 0xee, 0x3b, 0x00, 0x2f, 0x68, 0x3a, 0x53, 0xe7, 0xbc,
];
const PSS_EMPTY_SALT: [u8; 256] = [
    0x7e, 0x16, 0x81, 0x78, 0x6c, 0x1d, 0x03, 0xc1, 0xb1, 0x51, 0x95, 0xbd, 0xc8, 0x0f, 0x71, 0x62,
    0xc0, 0xae, 0x01, 0x2f, 0x7d, 0xf1, 0xe6, 0x5d, 0x38, 0x20, 0x97, 0xf0, 0xbd, 0x1b, 0x73, 0x36,
    0x8e, 0xe2, 0x81, 0xd4, 0x7b, 0xc2, 0x77, 0xbf, 0xc9, 0xe4, 0x9a, 0x8d, 0xc6, 0xb8, 0x58, 0x31,
    0x63, 0xea, 0x8a, 0x6b, 0xd8, 0x32, 0x65, 0x9b, 0xef, 0x4f, 0xa8, 0x61, 0xd8, 0x2c, 0xec, 0x2d,
    0xce, 0x76, 0x53, 0x0c, 0xe4, 0xa2, 0xc2, 0x42, 0xd4, 0x7a, 0xdd, 0x9f, 0x31, 0x6a, 0xcd, 0x37,
    0x2c, 0xf0, 0xa1, 0xd6, 0xe2, 0xb7, 0x5b, 0xfa, 0x70, 0xce, 0xaa, 0x48, 0xd8, 0x2f, 0x92, 0x94,
    0x59, 0xf3, 0x43, 0x56, 0x2b, 0x11, 0x1e, 0xa5, 0x4c, 0x73, 0xf8, 0x0c, 0x61, 0x56, 0xcd, 0x74,
    0x35, 0xe3, 0x36, 0x9b, 0x33, 0x0d, 0x61, 0x4a, 0x9b, 0x0e, 0x2f, 0x5a, 0x59, 0xe1, 0x4e, 0x98,
    0xd4, 0x8e, 0x4e, 0x44, 0xb6, 0x78, 0xab, 0xa3, 0xb4, 0xa3, 0xb1, 0xa0, 0x1d, 0xfd, 0x8b, 0xe3,
    0xf7, 0xd1, 0x3f, 0x8a, 0x9e, 0xd5, 0x64, 0xe3, 0x9f, 0xa2, 0xb2, 0x93, 0xfb, 0x9e, 0xf0, 0xff,
    0xc7, 0xd5, 0x30, 0xf9, 0x61, 0x3a, 0x61, 0x31, 0x08, 0x9b, 0x5b, 0x21, 0xa9, 0xa9, 0xc5, 0x74,
    0x51, 0xb2, 0x11, 0x82, 0x5a, 0x65, 0xe4, 0xbb, 0xe5, 0xa1, 0xad, 0x88, 0x52, 0x08, 0xad, 0xa9,
    0x04, 0x67, 0x91, 0x08, 0x61, 0x6e, 0xdb, 0x89, 0xa0, 0x4c, 0x2e, 0x63, 0x8a, 0x26, 0xb3, 0x47,
    0x07, 0x91, 0x5d, 0xa8, 0xe5, 0x63, 0x15, 0xd8, 0xe2, 0x38, 0xf0, 0x65, 0xdd, 0xc9, 0xfe, 0xfe,
    0x07, 0x70, 0x60, 0xb3, 0x40, 0x98, 0xaa, 0x56, 0xa9, 0xe3, 0x82, 0x1a, 0xa5, 0xb0, 0xbd, 0xca,
    0x54, 0x62, 0xde, 0x9c, 0x10, 0x09, 0x00, 0x40, 0xee, 0xeb, 0x4f, 0x64, 0x08, 0x48, 0xda, 0xbc,
];
const PSS_MAX_SALT: [u8; 256] = [
    0x16, 0x10, 0x63, 0xf5, 0x58, 0x06, 0x1c, 0x71, 0x9e, 0x24, 0xea, 0x4a, 0xc9, 0x17, 0x59, 0x6e,
    0x2f, 0x27, 0xc0, 0xe4, 0x16, 0x37, 0x49, 0xb2, 0xdb, 0x29, 0x43, 0x72, 0x33, 0x63, 0xff, 0xae,
    0x8d, 0x97, 0x2a, 0x5a, 0x1c, 0xf5, 0x68, 0x86, 0xd1, 0x0c, 0xd4, 0x5a, 0xc8, 0x11, 0x39, 0xe2,
    0x31, 0xf8, 0x1d, 0xa7, 0x24, 0x7f, 0x54, 0xa2, 0xa7, 0x00, 0x21, 0xdf, 0x42, 0x69, 0x72, 0xdb,
    0x2f, 0xbb, 0xd5, 0x9a, 0xf4, 0xb1, 0xf4, 0xd3, 0x3d, 0x1d, 0x12, 0xc2, 0xfc, 0xcb, 0xb9, 0x64,
    0x8d, 0x5f, 0x27, 0xda, 0x59, 0x98, 0x58, 0xcc, 0x0d, 0x43, 0x5b, 0x51, 0xcc, 0xd3, 0x55, 0x2a,
    0xa2, 0x06, 0x32, 0xbe, 0x70, 0x57, 0x81, 0x39, 0xbd, 0x79, 0x55, 0x86, 0xe7, 0xde, 0xa0, 0xe4,
    0x86, 0xa4, 0x0f, 0x35, 0x9d, 0x0d, 0x0c, 0x52, 0x4b, 0x50, 0x7d, 0xb5, 0x5a, 0x51, 0x33, 0xcb,
    0x5f, 0x95, 0xe4, 0xe0, 0x0a, 0x6f, 0xe0, 0xd6, 0x2e, 0xb2, 0xf9, 0xc3, 0xb2, 0xff, 0xf8, 0xaf,
    0xde, 0xed, 0xc2, 0x33, 0x93, 0x87, 0xac, 0x76, 0x16, 0xbc, 0x54, 0x90, 0x3f, 0x26, 0x4a, 0xd4,
    0x93, 0xc1, 0x19, 0xae, 0x1d, 0x08, 0x9e, 0x87, 0x32, 0xef, 0xe6, 0x7a, 0x1d, 0xae, 0x5e, 0x03,
    0x62, 0xae, 0x6f, 0x69, 0x42, 0x5d, 0x70, 0x9d, 0x41, 0x04, 0xd1, 0x8e, 0xc4, 0xc8, 0x39, 0x09,
    0xe9, 0xe5, 0x5a, 0x60, 0x0a, 0x7f, 0x94, 0x8c, 0x89, 0x43, 0x12, 0xb9, 0x56, 0x13, 0x22, 0x59,
    0x76, 0xd1, 0x10, 0xc3, 0xab, 0x72, 0xdd, 0x5b, 0xf6, 0xbb, 0x0e, 0x3d, 0xa4, 0x78, 0x2a, 0x9d,
    0x58, 0x38, 0x9e, 0x91, 0x7d, 0x3a, 0x7d, 0x20, 0xda, 0x94, 0xec, 0x9f, 0x39, 0xc7, 0x7c, 0x17,
    0xb1, 0x8a, 0xc7, 0x2f, 0x65, 0x15, 0x69, 0x0b, 0x77, 0xd0, 0xa2, 0x39, 0xb5, 0x6e, 0x76, 0xbc,
];
const RSA_SIGNATURE_65: [u8; 256] = [
    145, 162, 206, 34, 122, 227, 57, 37, 205, 20, 210, 159, 102, 220, 164, 49, 178, 249, 56, 23,
    108, 141, 192, 216, 87, 218, 146, 161, 191, 146, 9, 153, 51, 173, 32, 128, 181, 117, 63, 70,
    98, 219, 8, 116, 155, 160, 27, 37, 225, 192, 236, 69, 41, 22, 125, 166, 12, 153, 33, 45, 13,
    54, 217, 29, 139, 151, 105, 11, 20, 186, 212, 47, 139, 147, 164, 171, 193, 181, 96, 75, 2, 115,
    165, 207, 192, 75, 235, 124, 3, 113, 225, 75, 221, 179, 103, 100, 65, 42, 191, 24, 7, 243, 113,
    92, 250, 230, 209, 139, 78, 106, 232, 65, 29, 168, 20, 221, 196, 66, 192, 140, 13, 184, 218,
    223, 232, 25, 182, 213, 232, 253, 176, 80, 151, 150, 22, 23, 155, 237, 205, 73, 9, 235, 71, 39,
    230, 244, 57, 71, 169, 112, 4, 63, 210, 197, 46, 177, 50, 35, 193, 92, 10, 41, 75, 215, 211,
    221, 25, 238, 128, 111, 224, 239, 82, 147, 61, 121, 193, 223, 108, 6, 27, 173, 113, 140, 106,
    67, 123, 105, 250, 33, 81, 128, 150, 87, 228, 225, 189, 49, 226, 241, 56, 46, 252, 228, 113,
    163, 149, 118, 160, 25, 152, 151, 82, 104, 54, 142, 210, 240, 63, 32, 115, 71, 171, 178, 82,
    93, 186, 36, 150, 147, 203, 110, 44, 213, 164, 45, 195, 32, 86, 98, 57, 243, 221, 155, 51, 157,
    196, 85, 170, 47, 103, 4, 91, 144, 73, 87,
];
const COMPOSITE_DP: [u8; 128] = [
    0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
    0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
    0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
    0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
    0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
    0, 0, 255, 255, 0, 0, 255, 255,
];
const COMPOSITE_Q_INV: [u8; 128] = [
    232, 113, 37, 66, 119, 169, 161, 25, 107, 74, 24, 49, 150, 186, 13, 197, 215, 32, 170, 74, 96,
    137, 201, 28, 149, 174, 221, 216, 83, 92, 118, 97, 156, 90, 233, 36, 38, 222, 96, 32, 197, 110,
    59, 3, 161, 229, 44, 208, 95, 163, 19, 94, 74, 101, 155, 115, 152, 32, 68, 151, 242, 148, 31,
    234, 254, 232, 140, 21, 158, 177, 246, 99, 114, 114, 139, 253, 42, 34, 110, 7, 54, 26, 135,
    208, 33, 26, 251, 197, 223, 25, 214, 246, 103, 89, 227, 202, 144, 2, 237, 213, 31, 139, 40,
    109, 130, 51, 246, 225, 12, 88, 122, 61, 234, 96, 200, 254, 18, 235, 170, 64, 53, 84, 183, 204,
    57, 138, 89, 177,
];

fn crt_key() -> Rsa2048Crt {
    Rsa2048Crt::new(
        &black_box(RSA_P),
        &black_box(RSA_Q),
        &black_box(RSA_DP),
        &black_box(RSA_DQ),
        &black_box(RSA_Q_INV),
    )
    .unwrap()
}

/// PSS-encodes the sample digest with this salt for a 2048-bit modulus.
fn encoded(salt: &[u8], expected: &[u8; 256]) -> bool {
    let mut out = [0; 256];
    encode_sha256(&black_box(SAMPLE_SHA256), salt, 2047, &mut out).is_ok() && out == *expected
}

fn digest() -> [u8; 32] {
    let mut h = Sha256::new();
    let message = black_box(b"header\0\0footer");
    for source in [0, 1] {
        for length in [0, 1, 32, 222] {
            let request = black_box(RsaPssRequest {
                p: RSA_P,
                q: RSA_Q,
                dp: RSA_DP,
                dq: RSA_DQ,
                q_inv: RSA_Q_INV,
                digest: Sha256::digest(message),
                salt: [255; 222],
                reserved: [0; 2],
                offset: 6,
                length: 2,
                source,
                salt_length: length,
            });
            let pattern = black_box(HexPattern::new("", "", 256).unwrap());
            let count = if length == 0 && source == 0 { 1 } else { 4 };
            for counter in 0..count {
                record_candidate(
                    &mut h,
                    rsa_pss(&request, message, black_box(counter), &pattern),
                );
            }
        }
    }
    h.finalize()
}

checks! {
    /// rsa pss sha256
    fn sha256() -> bool {
        Sha256::digest(black_box(b"sample")) == SAMPLE_SHA256
    }

    /// rsa pss mgf1 partial block
    fn mgf1_partial_block() -> bool {
        let mut out = [0; 50];
        mgf1_sha256(black_box(b"public test seed"), &mut out).is_ok() && out == MGF_PARTIAL
    }

    /// rsa pss salt32 encoding
    fn salt32_encoding() -> bool {
        let salt = black_box(core::array::from_fn::<_, 32, _>(|i| i as u8));
        encoded(&salt, &PSS_SALT32)
    }

    /// rsa pss empty salt encoding
    fn empty_salt_encoding() -> bool {
        encoded(&black_box([0u8; 0]), &PSS_EMPTY_SALT)
    }

    /// rsa pss maximum salt encoding
    fn maximum_salt_encoding() -> bool {
        encoded(&black_box([0x42; 222]), &PSS_MAX_SALT)
    }

    /// rsa pss oversized salt rejected
    fn oversized_salt_rejected() -> bool {
        let mut out = [0xa5; 256];
        encode_sha256(&black_box(SAMPLE_SHA256), &black_box([0; 223]), 2047, &mut out)
            == Err(PssError::SaltTooLong)
            && out == [0xa5; 256]
    }

    /// rsa pss salt carry
    fn salt_carry() -> bool {
        let mut out = [0; 2];
        write_salt_counter(&black_box([0xff; 2]), black_box(1), &mut out).is_ok() && out == [0; 2]
    }

    /// rsa pss crt known answer
    fn crt_known_answer() -> bool {
        let mut input = [0; 256];
        input[255] = 65;
        crt_key().private_operation(&black_box(input)) == Some(RSA_SIGNATURE_65)
    }

    /// rsa pss crt fault rejected
    fn crt_fault_rejected() -> bool {
        // Deliberately composite p simulates inconsistent CRT arithmetic while
        // leaving constructor congruence checks satisfied. The final public-operation
        // verification must reject the result. No private fields or test hooks needed.
        let key = Rsa2048Crt::new(
            &black_box([255; 128]),
            &black_box(RSA_Q),
            &black_box(COMPOSITE_DP),
            &black_box(RSA_DQ),
            &black_box(COMPOSITE_Q_INV),
        )
        .unwrap();
        let mut input = [0; 256];
        input[255] = 65;
        key.private_operation(&black_box(input)).is_none()
    }

    /// rsa pss crt modulus rejected
    fn crt_modulus_rejected() -> bool {
        let key = crt_key();
        key.private_operation(&black_box(key.modulus())).is_none()
    }

    /// end-to-end rsa pss candidate pipeline
    fn end_to_end() -> bool {
        digest()
            == [
                186, 112, 218, 248, 118, 160, 144, 0, 191, 165, 67, 7, 165, 196, 6, 70, 218, 174,
                58, 193, 60, 84, 214, 84, 233, 131, 204, 111, 141, 86, 47, 85,
            ]
    }

    /// salt counter last value and exhaustion
    fn salt_counter_exhaustion() -> bool {
        let base = black_box([173]);
        let mut out = black_box([0xa5]);
        if write_salt_counter(&base, black_box(255), &mut out) != Ok(()) || out != [172] {
            return false;
        }
        out = black_box([0xa5]);
        write_salt_counter(&base, black_box(256), &mut out) == Err(WindowError::Exhausted)
            && out == [0xa5]
    }

    /// empty salt capacity and mismatched output length
    fn salt_empty_and_invalid() -> bool {
        let empty = black_box(&[] as &[u8]);
        let mut out = [0xa5; 1];
        write_salt_counter(empty, black_box(0), &mut []) == Ok(())
            && write_salt_counter(empty, black_box(1), &mut []) == Err(WindowError::Exhausted)
            && write_salt_counter(empty, black_box(0), &mut out) == Err(WindowError::InvalidBounds)
            && out == [0xa5]
    }

    /// salt carry beyond counter width
    fn salt_carry_beyond_u64() -> bool {
        let mut base = [0xff; 16];
        base[0] = 0x12;
        let mut out = black_box([0xa5; 16]);
        let mut expected = [0; 16];
        expected[0] = 0x13;
        write_salt_counter(&black_box(base), black_box(1), &mut out) == Ok(()) && out == expected
    }
}
