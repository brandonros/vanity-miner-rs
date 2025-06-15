/*use k256::SecretKey;
use k256::elliptic_curve::sec1::ToEncodedPoint;

#[derive(Debug)]
pub enum Error {
    InvalidSecretKey,
}

impl From<k256::elliptic_curve::Error> for Error {
    fn from(_: k256::elliptic_curve::Error) -> Self {
        Error::InvalidSecretKey
    }
}

pub fn secp256k1_derive_public_key(private_key_bytes: &[u8; 32]) -> Result<[u8; 33], Error> {
    // Create secret key from bytes (validates it's in valid range)
    let secret_key = SecretKey::from_bytes(private_key_bytes.into())?;
    
    // Derive public key
    let public_key = secret_key.public_key();
    
    // Get compressed point (33 bytes: 0x02/0x03 prefix + 32 bytes x-coordinate)
    let encoded_point = public_key.to_encoded_point(true); // true = compressed
    let compressed_bytes = encoded_point.as_bytes();
    
    // Convert to fixed-size array
    let mut result = [0u8; 33];
    result.copy_from_slice(compressed_bytes);
    
    Ok(result)
}

pub fn secp256k1_derive_public_key_uncompressed(private_key_bytes: &[u8; 32]) -> Result<[u8; 65], Error> {
    // Create secret key from bytes
    let secret_key = SecretKey::from_bytes(private_key_bytes.into())?;
    
    // Derive public key
    let public_key = secret_key.public_key();
    
    // Get uncompressed point (65 bytes: 0x04 prefix + 32 bytes x + 32 bytes y)
    let encoded_point = public_key.to_encoded_point(false); // false = uncompressed
    let uncompressed_bytes = encoded_point.as_bytes();
    
    // Convert to fixed-size array
    let mut result = [0u8; 65];
    result.copy_from_slice(uncompressed_bytes);
    
    Ok(result)
}*/

use alloc::boxed::Box;

pub fn secp256k1_derive_public_key(private_key_bytes: &[u8; 32]) -> Result<[u8; 33], Box<dyn core::error::Error>> {
    todo!()
}

pub fn secp256k1_derive_public_key_uncompressed(private_key_bytes: &[u8; 32]) -> Result<[u8; 65], Box<dyn core::error::Error>> {
    todo!()
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
