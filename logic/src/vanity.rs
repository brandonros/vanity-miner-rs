pub struct VanityKeyRequest<'a> {
    pub prefix: &'a [u8],
    pub suffix: &'a [u8],
    pub thread_idx: usize,
    pub rng_seed: u64,
}

#[allow(dead_code)]
pub struct VanityKeyResult {
    pub private_key: [u8; 32],
    pub hashed_private_key: [u8; 64],
    pub public_key: [u8; 32],
    pub encoded_public_key: [u8; 64],
    pub encoded_len: usize,
    pub matches: bool,
}

/// Pure function for matching logic
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
pub fn generate_and_check_vanity_key(request: &VanityKeyRequest) -> VanityKeyResult {
    // Generate private key
    let private_key = crate::xoroshiro::generate_random_private_key(
        request.thread_idx, 
        request.rng_seed
    );
    
    // Hash private key
    let hashed_private_key = crate::sha512::sha512_32bytes_from_bytes(&private_key);
    
    // Derive public key
    let public_key = crate::ed25519::ed25519_derive_public_key(&hashed_private_key);
    
    // Encode public key
    let mut encoded_public_key = [0u8; 64];
    let encoded_len = crate::base58::base58_encode(&public_key, &mut encoded_public_key);
    
    // Check if matches vanity criteria
    let matches = check_vanity_match(&encoded_public_key, encoded_len, request.prefix, request.suffix);
    
    VanityKeyResult {
        private_key,
        hashed_private_key,
        public_key,
        encoded_public_key,
        encoded_len,
        matches,
    }
}
