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
    pub encoded_address: [u8; 42],      // hex encoded address with 0x prefix
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
pub fn generate_and_check_ethereum_vanity_key(request: &EthereumVanityKeyRequest) -> EthereumVanityKeyResult {
    todo!()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_generate_and_check_ethereum_vanity_key_correctly() {
       todo!()
    }
}
