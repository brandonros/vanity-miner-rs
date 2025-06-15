use super::error::Error;
use super::constants;

// Secret key wrapper
#[derive(Debug, Clone, Copy)]
pub struct SecretKey {
    pub data: [u8; 32],
}

impl SecretKey {
    pub fn from_slice(data: &[u8]) -> Result<Self, Error> {
        if data.len() != 32 {
            return Err(Error::InvalidSecretKey);
        }
        
        let mut key_data = [0u8; 32];
        key_data.copy_from_slice(data);
        
        // Validate that key is in range [1, n-1] where n is curve order
        if Self::is_zero(&key_data) || Self::is_ge_curve_order(&key_data) {
            return Err(Error::InvalidSecretKey);
        }
        
        Ok(Self { data: key_data })
    }
    
    fn is_zero(data: &[u8; 32]) -> bool {
        data.iter().all(|&b| b == 0)
    }
    
    fn is_ge_curve_order(data: &[u8; 32]) -> bool {
        // Compare with curve order
        for i in 0..32 {
            match data[i].cmp(&constants::CURVE_ORDER[i]) {
                core::cmp::Ordering::Less => return false,
                core::cmp::Ordering::Greater => return true,
                core::cmp::Ordering::Equal => continue,
            }
        }
        true // Equal to curve order, which is invalid
    }
}
