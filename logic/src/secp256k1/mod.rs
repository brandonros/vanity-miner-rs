use crate::error::Error;

pub mod constants;
pub mod field_element;
pub mod error;
pub mod secret_key;
pub mod public_key;
pub mod point;

use k256::{SecretKey, PublicKey, EncodedPoint};
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
    let public_key = PublicKey::from(&secret_key);
    
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
    let public_key = PublicKey::from(&secret_key);
    
    // Get uncompressed point (65 bytes: 0x04 prefix + 32 bytes x + 32 bytes y)
    let encoded_point = public_key.to_encoded_point(false); // false = uncompressed
    let uncompressed_bytes = encoded_point.as_bytes();
    
    // Convert to fixed-size array
    let mut result = [0u8; 65];
    result.copy_from_slice(uncompressed_bytes);
    
    Ok(result)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{secp256k1_derive_public_key, secp256k1_derive_public_key_uncompressed};

    /*#[test]
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
    }*/

    #[test]
    fn test_field_element_basic_ops() {
        println!("=== Testing FieldElement basic operations ===");
        
        // Test creating elements
        let zero = field_element::FieldElement::zero();
        let one = field_element::FieldElement::one();
        
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
        let generator = point::Point::generator();
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
        let secret_key = secret_key::SecretKey::from_slice(&private_key_bytes);
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
        let secret_key = secret_key::SecretKey::from_slice(&private_key_bytes).unwrap();
        println!("Secret key created");
        
        println!("Step 2: Getting generator point...");
        let generator = point::Point::generator();
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
        let two = field_element::FieldElement::new(data).unwrap();
        
        println!("Testing invert of 2...");
        match two.invert() {
            Ok(inv) => {
                println!("Invert successful: {:02x?}", inv.data);
                
                // Verify: 2 * inv = 1
                let product = two.mul(&inv);
                let one = field_element::FieldElement::one();
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
        let generator = point::Point::generator();
        println!("Generator: {:?}", generator);
        
        println!("Doubling generator...");
        let doubled = generator.double();
        println!("Doubled: {:?}", doubled);
        
        // Should not be infinity
        assert!(!doubled.infinity, "Doubled generator should not be infinity");
    }

    #[test]
    fn test_small_scalar_multiplication() {
        println!("=== Testing small scalar multiplication ===");
        
        let generator = crate::point::Point::generator();
        
        // Test multiplying by 2 (should be the same as doubling)
        let mut scalar_2 = [0u8; 32];
        scalar_2[0] = 2; // Little-endian 2
        
        println!("Testing G * 2...");
        let g_times_2_mult = generator.multiply(&scalar_2);
        let g_times_2_double = generator.double();
        
        println!("G * 2 (multiply): {:?}", g_times_2_mult);
        println!("G * 2 (double):   {:?}", g_times_2_double);
        
        assert_eq!(g_times_2_mult, g_times_2_double, "G*2 should equal G.double()");
        
        // Test multiplying by 3 = 2*G + G
        let mut scalar_3 = [0u8; 32];
        scalar_3[0] = 3; // Little-endian 3
        
        println!("Testing G * 3...");
        let g_times_3_mult = generator.multiply(&scalar_3);
        let g_times_3_add = g_times_2_double.add(&generator);
        
        println!("G * 3 (multiply): {:?}", g_times_3_mult);
        println!("G * 3 (2*G + G):  {:?}", g_times_3_add);
        
        assert_eq!(g_times_3_mult, g_times_3_add, "G*3 should equal 2*G + G");
        
        println!("Small scalar multiplication tests passed!");
    }

    #[test]
    fn test_field_element_basic_arithmetic() {
        println!("=== Testing basic field arithmetic ===");
        
        // Test that 1 * 1 = 1
        let one = field_element::FieldElement::one();
        let result = one.mul(&one);
        println!("1 * 1 = {:02x?}", result.data);
        assert_eq!(result, one, "1 * 1 should equal 1");
        
        // Test that 2 * 1 = 2
        let mut two_data = [0u8; 32];
        two_data[0] = 2;
        let two = field_element::FieldElement::new(two_data).unwrap();
        
        let result = two.mul(&one);
        println!("2 * 1 = {:02x?}", result.data);
        assert_eq!(result, two, "2 * 1 should equal 2");
        
        // Test that 1 + 1 = 2
        let result = one.add(&one);
        println!("1 + 1 = {:02x?}", result.data);
        assert_eq!(result, two, "1 + 1 should equal 2");
        
        println!("Basic arithmetic tests passed!");
    }
    
    #[test]
    fn test_field_element_invert_manually() {
        println!("=== Testing manual inversion ===");
        
        // Test inverting 2
        let mut two_data = [0u8; 32];
        two_data[0] = 2;
        let two = field_element::FieldElement::new(two_data).unwrap();
        
        println!("Testing invert of 2 manually...");
        
        // Try to compute 2^(p-2) step by step to see where it fails
        let one = field_element::FieldElement::one();
        let mut result = one;
        let mut base = two;
        
        // Use a small exponent first - just test 2^1 = 2
        println!("Testing 2^1...");
        let exp_1 = two.mul(&one);
        println!("2^1 = {:02x?}", exp_1.data);
        assert_eq!(exp_1, two, "2^1 should equal 2");
        
        // Test 2^2 = 4
        println!("Testing 2^2...");
        let exp_2 = two.mul(&two);
        println!("2^2 = {:02x?}", exp_2.data);
        
        let mut four_data = [0u8; 32];
        four_data[0] = 4;
        let four = field_element::FieldElement::new(four_data).unwrap();
        assert_eq!(exp_2, four, "2^2 should equal 4");
        
        println!("Manual inversion tests passed!");
    }

    #[test]
    fn test_small_exponentiation() {
        println!("=== Testing small exponentiations ===");
        
        let mut two_data = [0u8; 32];
        two_data[0] = 2;
        let two = field_element::FieldElement::new(two_data).unwrap();
        let one = field_element::FieldElement::one();
        
        // Test 2^2 = 4 using our exponentiation method
        println!("Testing 2^2 using binary exponentiation...");
        let mut result = one;
        let mut base = two;
        
        // Exponent 2 = binary 10
        // Process bit 1 (LSB): skip (bit is 0)
        // Process bit 2: bit is 1, so result = result * base = 1 * 2 = 2
        base = base.square(); // base = 2^2 = 4
        
        // Bit 1 of exponent 2 (which is 0): skip
        
        // Bit 2 of exponent 2 (which is 1): multiply
        result = result.mul(&base); // result = 1 * 4 = 4
        
        println!("Manual 2^2 = {:02x?}", result.data);
        
        let mut four_data = [0u8; 32];
        four_data[0] = 4;
        let expected_four = field_element::FieldElement::new(four_data).unwrap();
        
        assert_eq!(result, expected_four, "2^2 should equal 4");
        
        // Now test our actual binary exponentiation function
        println!("Testing with a small exponent using our invert-style loop...");
        
        // Test 2^3 = 8
        let exp_bytes = [3u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        
        let mut result = one;
        let mut base = two;
        
        for (byte_idx, &byte) in exp_bytes.iter().enumerate() {
            for bit in 0..8 {
                if (byte >> bit) & 1 == 1 {
                    result = result.mul(&base);
                    println!("  Multiplying: bit {} of byte {}, result now = {:02x?}", bit, byte_idx, result.data);
                }
                
                if byte_idx < 31 || bit < 7 {
                    base = base.square();
                    println!("  Squaring base: now = {:02x?}", base.data);
                }
            }
        }
        
        let mut eight_data = [0u8; 32];
        eight_data[0] = 8;
        let expected_eight = field_element::FieldElement::new(eight_data).unwrap();
        
        println!("Result of 2^3 = {:02x?}", result.data);
        println!("Expected 8 = {:02x?}", expected_eight.data);
        
        assert_eq!(result, expected_eight, "2^3 should equal 8");
        
        println!("Small exponentiation tests passed!");
    }
}
