use crate::base58;
use crate::secp256k1;
use crate::sha256;
use crate::ripemd160;
use crate::xoroshiro;

pub struct BitcoinVanityKeyRequest<'a> {
    pub prefix: &'a [u8],
    pub suffix: &'a [u8],
    pub thread_idx: usize,
    pub rng_seed: u64,
}

#[allow(dead_code)]
pub struct BitcoinVanityKeyResult {
    pub private_key: [u8; 32],
    pub public_key: [u8; 33],           // compressed secp256k1 public key
    pub public_key_hash: [u8; 20],      // RIPEMD160(SHA256(public_key))
    pub versioned_payload: [u8; 21],    // 0x00 + public_key_hash
    pub address_with_checksum: [u8; 25], // versioned_payload + 4-byte checksum
    pub encoded_address: [u8; 64],      // Base58 encoded address
    pub encoded_len: usize,
    pub matches: bool,
}

/// Pure function for matching logic - same as Solana
fn check_vanity_match(
    encoded_key: &[u8; 64],
    encoded_len: usize,
    prefix: &[u8],
    suffix: &[u8],
) -> bool {
    // Check prefix
    if prefix.len() > encoded_len {
        return false;
    }
    for i in 0..prefix.len() {
        if encoded_key[i] != prefix[i] {
            return false;
        }
    }
    
    // Check suffix
    if suffix.len() > encoded_len {
        return false;
    }
    for i in 0..suffix.len() {
        if encoded_key[encoded_len - suffix.len() + i] != suffix[i] {
            return false;
        }
    }
    
    true
}

/// Pure function - no side effects, easily testable
pub fn generate_and_check_bitcoin_vanity_key(request: &BitcoinVanityKeyRequest) -> BitcoinVanityKeyResult {
    // Generate private key
    let private_key = xoroshiro::generate_random_private_key(
        request.thread_idx, 
        request.rng_seed
    );
    
    // Derive public key (compressed secp256k1)
    let public_key = secp256k1::secp256k1_derive_public_key(&private_key)
        .unwrap_or([0u8; 33]); // Handle error case
    
    // Hash public key: RIPEMD160(SHA256(public_key))
    let sha256_hash = sha256::sha256_from_bytes(&public_key);
    let public_key_hash = ripemd160::ripemd160_32bytes_from_bytes(&sha256_hash);
    
    // Add version byte (0x00 for mainnet P2PKH)
    let mut versioned_payload = [0u8; 21];
    versioned_payload[0] = 0x00; // Version byte
    versioned_payload[1..21].copy_from_slice(&public_key_hash);
    
    // Calculate checksum: first 4 bytes of SHA256(SHA256(versioned_payload))
    let checksum_hash = sha256::sha256_from_bytes(&sha256::sha256_from_bytes(&versioned_payload));
    
    // Combine versioned payload + checksum
    let mut address_with_checksum = [0u8; 25];
    address_with_checksum[0..21].copy_from_slice(&versioned_payload);
    address_with_checksum[21..25].copy_from_slice(&checksum_hash[0..4]);
    
    // Base58 encode the final address
    let mut encoded_address = [0u8; 64];
    let encoded_len = base58::base58_encode_25(&address_with_checksum, &mut encoded_address);
    
    // Check if matches vanity criteria
    let matches = check_vanity_match(&encoded_address, encoded_len, request.prefix, request.suffix);
    
    BitcoinVanityKeyResult {
        private_key,
        public_key,
        public_key_hash,
        versioned_payload,
        address_with_checksum,
        encoded_address,
        encoded_len,
        matches,
    }
}
