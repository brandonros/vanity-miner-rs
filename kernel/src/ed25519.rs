pub fn ed25519_clamp(hashed_private_key_bytes: &mut [u8; 32]) {
    hashed_private_key_bytes[0] &= 248;
    hashed_private_key_bytes[31] &= 63;
    hashed_private_key_bytes[31] |= 64;
}

pub fn ed25519_derive_public_key(hashed_private_key_bytes: &[u8; 32]) -> [u8; 32] {
    let scalar = curve25519_dalek::Scalar::from_bytes_mod_order(*hashed_private_key_bytes);
    let point = curve25519_dalek::constants::ED25519_BASEPOINT_TABLE * &scalar;
    let compressed_point = point.compress();
    let public_key_bytes = compressed_point.to_bytes();
    public_key_bytes
}
