use crate::base58;
use crate::ed25519;
use crate::sha512;
use crate::vanity;
use crate::xoroshiro;

pub struct SolanaVanityKeyRequest<'a> {
    pub prefix: &'a [u8],
    pub suffix: &'a [u8],
    pub thread_idx: usize,
    pub rng_seed: u64,
}

#[allow(dead_code)]
pub struct SolanaVanityKeyResult {
    pub private_key: [u8; 32],
    pub hashed_private_key: [u8; 64],
    pub public_key: [u8; 32],
    pub encoded_public_key: [u8; 64],
    pub encoded_len: usize,
    pub matches: bool,
}

/// Pure function - no side effects, easily testable
pub fn generate_and_check_solana_vanity_key(request: &SolanaVanityKeyRequest) -> SolanaVanityKeyResult {
    // Generate private key
    let private_key = xoroshiro::generate_random_private_key(
        request.thread_idx, 
        request.rng_seed
    );
    
    // Hash private key
    let hashed_private_key = sha512::sha512_32bytes_from_bytes(&private_key);
    
    // Derive public key
    let public_key = ed25519::ed25519_derive_public_key(&hashed_private_key);
    
    // Encode public key
    let mut encoded_public_key = [0u8; 64];
    let encoded_len = base58::base58_encode_32(&public_key, &mut encoded_public_key);
    
    // Check if matches vanity criteria
    let matches = vanity::check_vanity_match(&encoded_public_key[..encoded_len], request.prefix, request.suffix);
    
    SolanaVanityKeyResult {
        private_key,
        hashed_private_key,
        public_key,
        encoded_public_key,
        encoded_len,
        matches,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_generate_and_check_solana_vanity_key_correctly() {
        // Arrange
        let prefix = b"aa";  // Example prefix
        let suffix = b""; // Example suffix
        let request = SolanaVanityKeyRequest {
            prefix,
            suffix,
            thread_idx: 5,
            rng_seed: 8574998174529019819,
        };

        // Act
        let result = generate_and_check_solana_vanity_key(&request);

        // Assert
        assert_eq!(result.matches, true);
        assert_eq!(
            result.private_key, 
            <[u8; 32]>::try_from(
                hex::decode("d32ef33913a75aada4fc64d153de08338e169234f3432cc0294510df9fd0ccf8")
                    .unwrap()
                    .as_slice()
            ).unwrap()
        );
        assert_eq!(result.encoded_public_key[0..result.encoded_len], *b"aaLs2GEHDEajV3kgXsr7FPDRc4mcKVJLQDXnWWcyJCr");   
    }
}