use k256::SecretKey;
use k256::elliptic_curve::{point::AffineCoordinates, sec1::ToEncodedPoint};

/// Derive a compressed SEC1 public key, rejecting zero and out-of-range scalars.
pub fn try_secp256k1_derive_public_key(private_key_bytes: &[u8; 32]) -> Option<[u8; 33]> {
    let secret_key = SecretKey::from_bytes(private_key_bytes.into()).ok()?;
    let public_key = secret_key.public_key();
    let point = public_key.as_affine();
    let mut result = [0u8; 33];
    // SecretKey validation excludes the identity. Fixed SEC1 serialization avoids
    // the variable-length EncodedPoint identity/tag path on device compilers.
    result[0] = 2 | (point.y_is_odd().unwrap_u8() & 1);
    result[1..].copy_from_slice(&point.x());
    Some(result)
}

/// Derive an uncompressed SEC1 public key, rejecting invalid scalars.
pub fn try_secp256k1_derive_public_key_uncompressed(
    private_key_bytes: &[u8; 32],
) -> Option<[u8; 65]> {
    let secret_key = SecretKey::from_bytes(private_key_bytes.into()).ok()?;
    let public_key = secret_key.public_key();
    let encoded = public_key.to_encoded_point(false);
    encoded.as_bytes().try_into().ok()
}

pub fn secp256k1_derive_public_key(private_key_bytes: &[u8; 32]) -> [u8; 33] {
    try_secp256k1_derive_public_key(private_key_bytes).expect("invalid secp256k1 private key")
}

pub fn secp256k1_derive_public_key_uncompressed(private_key_bytes: &[u8; 32]) -> [u8; 65] {
    try_secp256k1_derive_public_key_uncompressed(private_key_bytes)
        .expect("invalid secp256k1 private key")
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        crypto::secp256k1::secp256k1_derive_public_key,
        crypto::secp256k1::secp256k1_derive_public_key_uncompressed,
    };

    #[test]
    fn checked_encodings_match_k256_and_reject_invalid_scalars() {
        for invalid in [
            [0u8; 32],
            [0xff; 32],
            hex::decode("fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141")
                .unwrap()
                .try_into()
                .unwrap(),
        ] {
            assert_eq!(try_secp256k1_derive_public_key(&invalid), None);
            assert_eq!(try_secp256k1_derive_public_key_uncompressed(&invalid), None);
        }
        let mut state = 0x123456789abcdef0u64;
        for _ in 0..32 {
            let mut input = [0u8; 32];
            for word in input.chunks_exact_mut(8) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                word.copy_from_slice(&state.to_be_bytes());
            }
            let key = k256::SecretKey::from_bytes((&input).into())
                .unwrap()
                .public_key();
            assert_eq!(
                try_secp256k1_derive_public_key(&input).unwrap().as_slice(),
                key.to_encoded_point(true).as_bytes()
            );
            assert_eq!(
                try_secp256k1_derive_public_key_uncompressed(&input)
                    .unwrap()
                    .as_slice(),
                key.to_encoded_point(false).as_bytes()
            );
        }
    }

    #[test]
    fn should_derive_compressed_public_key_correctly() {
        // Test vector from Bitcoin's secp256k1 implementation
        let private_key_bytes: [u8; 32] =
            hex::decode("152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb")
                .unwrap()
                .try_into()
                .unwrap();
        let public_key_bytes = secp256k1_derive_public_key(&private_key_bytes);

        // Expected compressed public key (33 bytes, starts with 0x02 or 0x03)
        let expected: [u8; 33] =
            hex::decode("039163ab449d4b90de13ce60b504bfc27a4aed378c1f8338686156b91445637c8d")
                .unwrap()
                .try_into()
                .unwrap();
        assert_eq!(public_key_bytes, expected);
    }

    #[test]
    fn should_derive_uncompressed_public_key_correctly() {
        // Same private key as above
        let private_key_bytes: [u8; 32] =
            hex::decode("152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb")
                .unwrap()
                .try_into()
                .unwrap();
        let public_key_bytes = secp256k1_derive_public_key_uncompressed(&private_key_bytes);

        // Expected uncompressed public key (65 bytes, starts with 0x04)
        let expected: [u8; 65] = hex::decode("049163ab449d4b90de13ce60b504bfc27a4aed378c1f8338686156b91445637c8d33272b79994dae54da4011cc3e3491ccdf3bd3fd92978a00873727f99beb4375").unwrap().try_into().unwrap();
        assert_eq!(public_key_bytes, expected);
    }
}
