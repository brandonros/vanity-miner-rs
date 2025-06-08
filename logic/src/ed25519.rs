pub fn ed25519_derive_public_key(hashed_private_key_bytes: &[u8; 64]) -> [u8; 32] {
    // copy only first 32 bytes
    let mut input = [0u8; 32];
    input.copy_from_slice(&hashed_private_key_bytes[0..32]);

    // clamp
    let clamped_input = curve25519_dalek::scalar::clamp_integer(input);

    // derive public key
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(clamped_input);
    let point = curve25519_dalek::constants::ED25519_BASEPOINT_TABLE * &scalar;
    let compressed_point = point.compress();
    let public_key_bytes = compressed_point.to_bytes();
    public_key_bytes
}


#[cfg(test)]
mod test {
    use crate::ed25519_derive_public_key;

    #[test]
    fn should_hash_correctly() {
        // derive public key
        let hashed_private_key_bytes: [u8; 64] = hex::decode("152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb0f5b8817a8e43510f83333417178f2f59fdc3c723199303a5f9be71af2f7b664").unwrap().try_into().unwrap();
        let public_key_bytes = ed25519_derive_public_key(&hashed_private_key_bytes);
        let expected = hex::decode("0af764c1b6133a3a0abd7ef9c853791b687ce1e235f9dc8466d886da314dbea7").unwrap();
        assert_eq!(public_key_bytes, *expected);
    }
}
