//! Candidate derivation shared by CPU and GPU. Seeds must come from OS entropy.
//!
//! Workers own distinct identifiers and monotonically increasing counters.
//! Domain separation makes PRF inputs disjoint; independently derived outputs
//! retain the negligible collision probability of a 256-bit random function.
//! Nothing in this module obtains entropy or prints secret material.

use crate::crypto::sha256::Sha256;
use hmac::{Mac, SimpleHmac};
use zeroize::Zeroize;

#[derive(Clone, Copy)]
pub enum CandidateDomain {
    P256PrivateKey,
    P256Ephemeral,
    RsaFactor,
    RsaRangeStart,
}

impl CandidateDomain {
    fn label(self) -> &'static [u8] {
        match self {
            Self::P256PrivateKey => b"p256-private-key",
            Self::P256Ephemeral => b"p256-ecdsa-ephemeral",
            Self::RsaFactor => b"rsa-factor",
            Self::RsaRangeStart => b"rsa-range-start",
        }
    }
}

/// Secret state deliberately has no Debug, Display, Clone, or serialization.
pub struct CandidateDeriver {
    seed: [u8; 32],
    domain: CandidateDomain,
    public_key_fingerprint: [u8; 32],
    message_digest: [u8; 32],
}

impl Drop for CandidateDeriver {
    fn drop(&mut self) {
        self.seed.zeroize();
    }
}

impl CandidateDeriver {
    /// Caller supplies fresh OS entropy for every search and clears its copy.
    /// Ephemeral searches must bind the actual public key and message digest.
    pub fn new(
        seed: [u8; 32],
        domain: CandidateDomain,
        public_key_fingerprint: [u8; 32],
        message_digest: [u8; 32],
    ) -> Self {
        Self {
            seed,
            domain,
            public_key_fingerprint,
            message_digest,
        }
    }

    /// `attempt` separates rejection-sampling retries for the same candidate.
    /// Consumers must reject counter overflow instead of restarting at zero.
    #[inline(always)]
    pub fn block(&self, worker: u64, counter: u128, attempt: u32) -> [u8; 32] {
        // HMAC-SHA256 accepts keys of every length, including this fixed 32 bytes.
        let mut mac = SimpleHmac::<Sha256>::new_from_slice(&self.seed)
            .expect("HMAC-SHA256 accepts 32-byte keys");
        mac.update(b"vanity-miner/crypto-search/v1\0");
        mac.update(self.domain.label());
        mac.update(&[0]);
        mac.update(&self.public_key_fingerprint);
        mac.update(&self.message_digest);
        mac.update(&worker.to_be_bytes());
        mac.update(&counter.to_be_bytes());
        mac.update(&attempt.to_be_bytes());
        mac.finalize().into_bytes().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn derivation_domains_are_distinct() {
        // Public synthetic seed only.
        let d = CandidateDeriver::new([0x42; 32], CandidateDomain::P256Ephemeral, [1; 32], [2; 32]);
        let reference = d.block(3, 4, 5);
        // Independently computed with Python's hmac/hashlib implementation.
        assert_eq!(
            hex::encode(reference),
            "779c976e69d7164af481530f0fc22b278cdc238e083535a6dd189942dc84e0aa"
        );
        assert_eq!(reference, d.block(3, 4, 5));
        for other in [d.block(4, 4, 5), d.block(3, 5, 5), d.block(3, 4, 6)] {
            assert_ne!(reference, other);
        }
        for other in [
            CandidateDeriver::new([0x43; 32], CandidateDomain::P256Ephemeral, [1; 32], [2; 32]),
            CandidateDeriver::new(
                [0x42; 32],
                CandidateDomain::P256PrivateKey,
                [1; 32],
                [2; 32],
            ),
            CandidateDeriver::new([0x42; 32], CandidateDomain::P256Ephemeral, [2; 32], [2; 32]),
            CandidateDeriver::new([0x42; 32], CandidateDomain::P256Ephemeral, [1; 32], [3; 32]),
        ] {
            assert_ne!(reference, other.block(3, 4, 5));
        }
    }
}
