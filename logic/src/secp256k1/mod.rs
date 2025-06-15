use crate::error::Error;

pub mod constants;
pub mod field_element;
pub mod error;
pub mod secret_key;
pub mod public_key;
pub mod point;

// Your existing functions using the new implementation
pub fn secp256k1_derive_public_key(private_key_bytes: &[u8; 32]) -> Result<[u8; 33], Error> {    
    // Create secret key (validates it's in valid range)
    let secret_key = secret_key::SecretKey::from_slice(private_key_bytes)?;
    
    // Derive public key
    let public_key = public_key::PublicKey::from_secret_key(&secret_key);
    
    // Return compressed format (33 bytes)
    Ok(public_key.serialize())
}

// For uncompressed format (65 bytes) - needed for Ethereum
pub fn secp256k1_derive_public_key_uncompressed(private_key_bytes: &[u8; 32]) -> Result<[u8; 65], Error> {
    let secret_key = secret_key::SecretKey::from_slice(private_key_bytes)?;
    let public_key = public_key::PublicKey::from_secret_key(&secret_key);
    
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

// Add this test to your mod.rs or create a separate test file

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::field_element::FieldElement;
    
    #[test]
    fn test_field_element_basic_ops() {
        println!("=== Testing FieldElement basic operations ===");
        
        // Test creating elements
        let zero = FieldElement::zero();
        let one = FieldElement::one();
        
        println!("Zero: {:?}", zero);
        println!("One: {:?}", one);
        
        // Test simple addition (this might be where it hangs)
        println!("Testing addition...");
        let result = one.add(&one);
        println!("1 + 1 = {:?}", result);
        
        // If we get here, addition works
        println!("Addition test passed!");
    }
    
    #[test]
    fn test_point_creation() {
        println!("=== Testing Point creation ===");
        
        // Try to create generator point (this uses constants and field ops)
        println!("Creating generator point...");
        let generator = crate::point::Point::generator();
        println!("Generator created: {:?}", generator);
    }
    
    #[test]
    fn test_secret_key_creation() {
        println!("=== Testing SecretKey creation ===");
        
        // Test with the same key from your failing test
        let private_key_bytes: [u8; 32] = [
            0x15, 0x2d, 0x53, 0x72, 0x3d, 0xa4, 0x20, 0x34,
            0x78, 0x57, 0x4b, 0x15, 0x31, 0x43, 0xa7, 0xea,
            0xa9, 0x21, 0xa8, 0xd8, 0x2c, 0x62, 0x95, 0x17,
            0xd6, 0xb1, 0x89, 0x49, 0xf0, 0x11, 0x1a, 0xbb,
        ];
        
        println!("Creating secret key...");
        let secret_key = crate::secret_key::SecretKey::from_slice(&private_key_bytes);
        match secret_key {
            Ok(sk) => println!("Secret key created successfully: {:?}", sk),
            Err(e) => println!("Secret key creation failed: {:?}", e),
        }
    }
    
    #[test]
    fn test_step_by_step_public_key_derivation() {
        println!("=== Testing step-by-step public key derivation ===");
        
        let private_key_bytes: [u8; 32] = [
            0x15, 0x2d, 0x53, 0x72, 0x3d, 0xa4, 0x20, 0x34,
            0x78, 0x57, 0x4b, 0x15, 0x31, 0x43, 0xa7, 0xea,
            0xa9, 0x21, 0xa8, 0xd8, 0x2c, 0x62, 0x95, 0x17,
            0xd6, 0xb1, 0x89, 0x49, 0xf0, 0x11, 0x1a, 0xbb,
        ];
        
        println!("Step 1: Creating secret key...");
        let secret_key = crate::secret_key::SecretKey::from_slice(&private_key_bytes).unwrap();
        println!("Secret key created");
        
        println!("Step 2: Getting generator point...");
        let generator = crate::point::Point::generator();
        println!("Generator point created");
        
        println!("Step 3: Performing scalar multiplication (this is likely where it hangs)...");
        let public_point = generator.multiply(&secret_key.data);
        println!("Scalar multiplication completed!");
        
        println!("Step 4: Serializing public key...");
        let public_key_bytes = public_point.compress();
        println!("Public key: {:02x?}", public_key_bytes);
    }
    
    #[test]
    fn test_field_element_invert() {
        println!("=== Testing FieldElement invert ===");
        
        // Test inverting a simple non-zero element
        let mut data = [0u8; 32];
        data[0] = 2; // Little-endian 2
        let two = FieldElement::new(data).unwrap();
        
        println!("Testing invert of 2...");
        match two.invert() {
            Ok(inv) => {
                println!("Invert successful: {:02x?}", inv.data);
                
                // Verify: 2 * inv = 1
                let product = two.mul(&inv);
                let one = FieldElement::one();
                println!("2 * inv = {:02x?}", product.data);
                println!("1 = {:02x?}", one.data);
                assert_eq!(product, one, "2 * inv should equal 1");
            },
            Err(e) => {
                println!("Invert failed: {:?}", e);
                panic!("Should be able to invert 2");
            }
        }
    }
    
    #[test]
    fn test_point_doubling_simple() {
        println!("=== Testing Point doubling ===");
        
        // Test doubling the generator point
        let generator = crate::point::Point::generator();
        println!("Generator: {:?}", generator);
        
        println!("Doubling generator...");
        let doubled = generator.double();
        println!("Doubled: {:?}", doubled);
        
        // Should not be infinity
        assert!(!doubled.infinity, "Doubled generator should not be infinity");
    }
}
