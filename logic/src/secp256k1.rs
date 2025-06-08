use secp256k1::{Secp256k1, SecretKey, PublicKey};

pub fn secp256k1_derive_public_key(private_key_bytes: &[u8; 32]) -> Result<[u8; 33], secp256k1::Error> {
    let secp = Secp256k1::new();
    
    // Create secret key (validates it's in valid range)
    let secret_key = SecretKey::from_slice(private_key_bytes)?;
    
    // Derive public key
    let public_key = PublicKey::from_secret_key(&secp, &secret_key);
    
    // Return compressed format (33 bytes)
    Ok(public_key.serialize())
}

// For uncompressed format (65 bytes) - needed for Ethereum
pub fn secp256k1_derive_public_key_uncompressed(private_key_bytes: &[u8; 32]) -> Result<[u8; 65], secp256k1::Error> {
    let secp = Secp256k1::new();
    let secret_key = SecretKey::from_slice(private_key_bytes)?;
    let public_key = PublicKey::from_secret_key(&secp, &secret_key);
    
    // Return uncompressed format
    Ok(public_key.serialize_uncompressed())
}

#[cfg(test)]
mod test {
    use crate::{secp256k1_derive_public_key, secp256k1_derive_public_key_uncompressed};

    #[test]
    fn should_derive_compressed_public_key_correctly() {
        // Test vector from Bitcoin's secp256k1 implementation
        let private_key_bytes: [u8; 32] = hex::decode("152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb").unwrap().try_into().unwrap();
        let public_key_bytes = secp256k1_derive_public_key(&private_key_bytes).unwrap();
        
        // Expected compressed public key (33 bytes, starts with 0x02 or 0x03)
        let expected: [u8; 33] = hex::decode("039163ab449d4b90de13ce60b504bfc27a4aed378c1f8338686156b91445637c8d").unwrap().try_into().unwrap();
        assert_eq!(public_key_bytes, expected);
    }

    #[test]
    fn should_derive_uncompressed_public_key_correctly() {
        // Same private key as above
        let private_key_bytes: [u8; 32] = hex::decode("152d53723da4203478574b153143a7eaa921a8d82c629517d6b18949f0111abb").unwrap().try_into().unwrap();
        let public_key_bytes = secp256k1_derive_public_key_uncompressed(&private_key_bytes).unwrap();
        
        // Expected uncompressed public key (65 bytes, starts with 0x04)
        let expected: [u8; 65] = hex::decode("049163ab449d4b90de13ce60b504bfc27a4aed378c1f8338686156b91445637c8d33272b79994dae54da4011cc3e3491ccdf3bd3fd92978a00873727f99beb4375").unwrap().try_into().unwrap();
        assert_eq!(public_key_bytes, expected);
    }
}