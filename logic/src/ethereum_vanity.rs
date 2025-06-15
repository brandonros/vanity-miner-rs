use crate::secp256k1;
use crate::keccak256;
use crate::xoroshiro;

pub struct EthereumVanityKeyRequest<'a> {
    pub prefix: &'a [u8],
    pub suffix: &'a [u8],
    pub thread_idx: usize,
    pub rng_seed: u64,
}

#[allow(dead_code)]
pub struct EthereumVanityKeyResult {
    pub private_key: [u8; 32],
    pub public_key: [u8; 64],           // uncompressed secp256k1 public key (without 0x04 prefix)
    pub address: [u8; 20],              // last 20 bytes of keccak256(public_key)
    pub matches: bool,
}

/// Pure function for matching logic - same as Solana
fn check_vanity_match(
    address: &[u8; 20],
    prefix: &[u8],
    suffix: &[u8],
) -> bool {
    let address_len = 20;
    
    // Check prefix
    if prefix.len() > address_len {
        return false;
    }
    for i in 0..prefix.len() {
        if address[i] != prefix[i] {
            return false;
        }
    }
    
    // Check suffix
    if suffix.len() > address_len {
        return false;
    }
    for i in 0..suffix.len() {
        if address[address_len - suffix.len() + i] != suffix[i] {
            return false;
        }
    }
    
    true
}

/// Pure function - no side effects, easily testable
pub fn generate_and_check_ethereum_vanity_key(request: &EthereumVanityKeyRequest) -> EthereumVanityKeyResult {
    // Generate private key
    let private_key = xoroshiro::generate_random_private_key(
        request.thread_idx, 
        request.rng_seed
    );
    
    // Derive uncompressed public key (65 bytes total, but we skip the 0x04 prefix)
    let full_public_key = secp256k1::secp256k1_derive_public_key_uncompressed(&private_key)
        .unwrap_or([0u8; 65]); // Handle error case
    
    // Extract the 64-byte public key (skip 0x04 prefix)
    let mut public_key = [0u8; 64];
    public_key.copy_from_slice(&full_public_key[1..]);
    
    // Calculate Ethereum address: keccak256(public_key)[12..]
    let public_key_hash = keccak256::keccak256_64bytes(&public_key);
    let mut address = [0u8; 20];
    address.copy_from_slice(&public_key_hash[12..]);
    
    // Check if matches vanity criteria (no hex encoding needed!)
    let matches = check_vanity_match(&address, request.prefix, request.suffix);
    
    EthereumVanityKeyResult {
        private_key,
        public_key,
        address,
        matches,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_generate_and_check_ethereum_vanity_key_correctly() {
        // Arrange
        let prefix = &[0x55];  // Example prefix
        let suffix = &[]; // Example suffix
        let request = EthereumVanityKeyRequest {
            prefix,
            suffix,
            thread_idx: 0,
            rng_seed: 10088153575472065218,
        };

        // Act
        let result = generate_and_check_ethereum_vanity_key(&request);

        // Assert
        assert_eq!(
            result.private_key, 
            <[u8; 32]>::try_from(
                hex::decode("23a33f35737ab1abc16cc1d17555c8dc751833ac76cf4bc9e32faf3d7352e930")
                    .unwrap()
                    .as_slice()
            ).unwrap()
        );
        assert_eq!(
            result.public_key, 
            <[u8; 64]>::try_from(
                hex::decode("bd954ff18736033d7eb34a760a16e7096a0f3a00e74f541d17e55619e6510b16cd17b6e2f30c2c1ca82a935229969125c4b075ced6c7d43c2175f58329009425")
                    .unwrap()
                    .as_slice()
            ).unwrap()
        );
        assert_eq!(
            result.address, 
            <[u8; 20]>::try_from(
                hex::decode("55e56b7b70dc37a7a1419e1e84ea4e6e237ef602")
                    .unwrap()
                    .as_slice()
            ).unwrap()
        );
        assert_eq!(result.matches, true);
    }
}
