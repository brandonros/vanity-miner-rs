//! p256 public key: key derivation, point arithmetic, scalar bounds,
//! coordinate encoding, hex-pattern matching, and the candidate pipeline.
use super::{P256_GENERATOR, record_candidate};
use crate::crypto::p256::{PublicTarget, candidate_scalar, public_point};
use crate::crypto::sha256::Sha256;
use crate::modes::p256_public_key::{P256PublicRequest, p256_public};
use crate::search::candidate_derivation::{CandidateDeriver, CandidateDomain};
use crate::search::hex_pattern::HexPattern;
use core::hint::black_box;

const P256_ORDER: [u8; 32] = [
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xbc, 0xe6, 0xfa, 0xad, 0xa7, 0x17, 0x9e, 0x84, 0xf3, 0xb9, 0xca, 0xc2, 0xfc, 0x63, 0x25, 0x51,
];
const P256_DOUBLE: [u8; 65] = [
    0x04, 0x7c, 0xf2, 0x7b, 0x18, 0x8d, 0x03, 0x4f, 0x7e, 0x8a, 0x52, 0x38, 0x03, 0x04, 0xb5, 0x1a,
    0xc3, 0xc0, 0x89, 0x69, 0xe2, 0x77, 0xf2, 0x1b, 0x35, 0xa6, 0x0b, 0x48, 0xfc, 0x47, 0x66, 0x99,
    0x78, 0x07, 0x77, 0x55, 0x10, 0xdb, 0x8e, 0xd0, 0x40, 0x29, 0x3d, 0x9a, 0xc6, 0x9f, 0x74, 0x30,
    0xdb, 0xba, 0x7d, 0xad, 0xe6, 0x3c, 0xe9, 0x82, 0x29, 0x9e, 0x04, 0xb7, 0x9d, 0x22, 0x78, 0x73,
    0xd1,
];
const PRIVATE_DERIVATION: [u8; 32] = [
    0x60, 0xb3, 0xb8, 0x4d, 0xdd, 0x39, 0x30, 0x08, 0x17, 0x6a, 0xa7, 0x6e, 0xf6, 0x29, 0x99, 0xa4,
    0xe8, 0x6e, 0xb9, 0x59, 0x81, 0x1f, 0x91, 0xf4, 0x4d, 0xca, 0x64, 0x8e, 0xae, 0xf1, 0xfe, 0xd6,
];
const NEGATIVE_GENERATOR: [u8; 65] = [
    0x04, 0x6b, 0x17, 0xd1, 0xf2, 0xe1, 0x2c, 0x42, 0x47, 0xf8, 0xbc, 0xe6, 0xe5, 0x63, 0xa4, 0x40,
    0xf2, 0x77, 0x03, 0x7d, 0x81, 0x2d, 0xeb, 0x33, 0xa0, 0xf4, 0xa1, 0x39, 0x45, 0xd8, 0x98, 0xc2,
    0x96, 0xb0, 0x1c, 0xbd, 0x1c, 0x01, 0xe5, 0x80, 0x65, 0x71, 0x18, 0x14, 0xb5, 0x83, 0xf0, 0x61,
    0xe9, 0xd4, 0x31, 0xcc, 0xa9, 0x94, 0xce, 0xa1, 0x31, 0x34, 0x49, 0xbf, 0x97, 0xc8, 0x40, 0xae,
    0x0a,
];

fn deriver() -> CandidateDeriver {
    CandidateDeriver::new(
        black_box([0x42; 32]),
        CandidateDomain::P256PrivateKey,
        [0; 32],
        [0; 32],
    )
}

fn digest() -> [u8; 32] {
    let mut h = Sha256::new();
    for (target, width) in [(0, 32), (1, 32), (2, 64), (3, 65)] {
        let request = black_box(P256PublicRequest {
            seed: [0x42; 32],
            worker: 7,
            target,
            reserved: 0,
        });
        for prefix in ["", "f"] {
            let pattern = black_box(HexPattern::new(prefix, "", width).unwrap());
            for counter in u64::MAX - 7..=u64::MAX {
                record_candidate(&mut h, p256_public(&request, black_box(counter), &pattern));
            }
        }
    }
    h.finalize()
}

checks! {
    /// p256 public key hmac derivation
    fn hmac_derivation() -> bool {
        deriver().block(black_box(3), black_box(4), 0) == PRIVATE_DERIVATION
    }

    /// p256 public key scalar derivation
    fn scalar_derivation() -> bool {
        candidate_scalar(&deriver(), black_box(3), black_box(4))
            .is_some_and(|s| *s == PRIVATE_DERIVATION)
    }

    /// p256 public key generator
    fn generator() -> bool {
        let mut scalar = [0; 32];
        scalar[31] = 1;
        public_point(&black_box(scalar)) == Some(P256_GENERATOR)
    }

    /// p256 public key point double
    fn point_double() -> bool {
        let mut scalar = [0; 32];
        scalar[31] = 2;
        public_point(&black_box(scalar)) == Some(P256_DOUBLE)
    }

    /// p256 public key zero scalar rejected
    fn zero_scalar_rejected() -> bool {
        public_point(&black_box([0; 32])).is_none()
    }

    /// p256 public key order scalar rejected
    fn order_scalar_rejected() -> bool {
        public_point(&black_box(P256_ORDER)).is_none()
    }

    /// p256 public key x encoding
    fn x_encoding() -> bool {
        PublicTarget::X.bytes(&black_box(P256_GENERATOR)) == &P256_GENERATOR[1..33]
    }

    /// p256 public key y encoding
    fn y_encoding() -> bool {
        PublicTarget::Y.bytes(&black_box(P256_GENERATOR)) == &P256_GENERATOR[33..65]
    }

    /// end-to-end p256 public candidate pipeline
    fn end_to_end() -> bool {
        digest()
            == [
                239, 76, 14, 206, 113, 184, 37, 233, 82, 252, 160, 108, 166, 113, 26, 230, 132, 89,
                110, 201, 235, 102, 51, 125, 141, 160, 109, 133, 191, 253, 11, 243,
            ]
    }

    /// hex pattern odd nibbles suffix and width
    fn hex_pattern_nibbles() -> bool {
        // Construction occurs on the host in production; black_box the resulting device record.
        let pattern = black_box(HexPattern::new("AbC", "dEf", 4).unwrap());
        pattern.matches(&black_box([0xab, 0xc0, 0x0d, 0xef]))
            && pattern.matches(&black_box([0xab, 0xcf, 0xfd, 0xef]))
            && !pattern.matches(&black_box([0xab, 0xbc, 0x0d, 0xef]))
            && !pattern.matches(&black_box([0xab, 0xc0, 0xd0, 0xef]))
            && !pattern.matches(&black_box([0xab, 0xc0, 0x0d, 0xef, 0]))
    }

    /// hex pattern maximum width checks the last byte and ignores free nibbles
    fn hex_pattern_max_width() -> bool {
        // The matcher is shared with RSA's 256-byte targets.
        let pattern = black_box(HexPattern::new("d", "f", 256).unwrap());
        let mut input = black_box([0xa5; 256]);
        input[0] = black_box(0xda);
        input[255] = black_box(0x1f);
        if !pattern.matches(&input) || pattern.matches(&input[..255]) {
            return false;
        }
        input[0] = black_box(0xd0);
        input[127] = black_box(0);
        input[255] = black_box(0xef);
        if !pattern.matches(&input) {
            return false;
        }
        input[255] = black_box(0xee);
        if pattern.matches(&input) {
            return false;
        }
        input[255] = black_box(0xef);
        input[0] = black_box(0xc0);
        !pattern.matches(&input)
    }

    /// p256 order minus one produces negative generator
    fn order_minus_one() -> bool {
        let mut scalar = P256_ORDER;
        scalar[31] -= 1;
        public_point(&black_box(scalar)) == Some(NEGATIVE_GENERATOR)
    }

    /// p256 order plus one rejected
    fn above_order() -> bool {
        let mut scalar = P256_ORDER;
        scalar[31] += 1;
        public_point(&black_box(scalar)).is_none()
    }
}
