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
