//! Candidate derivation shared by CPU and GPU. Seeds must come from OS entropy.
//!
//! Workers own distinct identifiers and monotonically increasing counters.
//! Domain separation makes PRF inputs disjoint; independently derived outputs
//! retain the negligible collision probability of a 256-bit random function.
//! Nothing in this module obtains entropy or prints secret material.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use zeroize::Zeroize;

#[derive(Clone, Copy)]
pub enum CandidateDomain {
    P256PrivateKey,
    P256Ephemeral,
    RsaPrime,
}

impl CandidateDomain {
    fn label(self) -> &'static [u8] {
        match self {
            Self::P256PrivateKey => b"p256-private-key",
            Self::P256Ephemeral => b"p256-ecdsa-ephemeral",
            Self::RsaPrime => b"rsa-2048-prime",
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
    pub fn block(&self, worker: u64, counter: u128, attempt: u32) -> [u8; 32] {
        // HMAC-SHA256 accepts keys of every length, including this fixed 32 bytes.
        let mut mac =
            Hmac::<Sha256>::new_from_slice(&self.seed).expect("HMAC-SHA256 accepts 32-byte keys");
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowError {
    InvalidBounds,
    Exhausted,
}

impl core::fmt::Display for WindowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidBounds => "mutable message window must be nonempty and within the message",
            Self::Exhausted => "candidate counter exceeds the message window capacity",
        })
    }
}

impl core::error::Error for WindowError {}

/// Enumerate fixed-width salts by adding a unique counter to a random starting
/// value modulo 2^(8*length). The complete finite space can be visited without
/// repetition; a zero-byte salt has exactly one candidate.
pub fn write_salt_counter(base: &[u8], counter: u64, output: &mut [u8]) -> Result<(), WindowError> {
    if base.len() != output.len() {
        return Err(WindowError::InvalidBounds);
    }
    if base.len() < 8 && counter >= (1u64 << (base.len() * 8)) {
        return Err(WindowError::Exhausted);
    }
    let mut carry = counter as u128;
    for i in (0..base.len()).rev() {
        let sum = base[i] as u128 + (carry & 255);
        output[i] = sum as u8;
        carry = (carry >> 8) + (sum >> 8);
    }
    Ok(())
}

/// Writes an injective big-endian counter encoding exclusively inside a window.
/// Runners partition the global counter space, for example by assigning disjoint
/// batches. Never use per-worker counters without a globally unique mapping.
pub fn write_message_counter(
    message: &mut [u8],
    offset: usize,
    length: usize,
    counter: u128,
) -> Result<(), WindowError> {
    let end = offset
        .checked_add(length)
        .ok_or(WindowError::InvalidBounds)?;
    if length == 0 || end > message.len() {
        return Err(WindowError::InvalidBounds);
    }
    if length < 16 && counter >= (1u128 << (length * 8)) {
        return Err(WindowError::Exhausted);
    }
    let bytes = counter.to_be_bytes();
    let count = length.min(bytes.len());
    message[offset..end].fill(0);
    message[end - count..end].copy_from_slice(&bytes[bytes.len() - count..]);
    Ok(())
}

/// Hash a virtual mutable window without allocating a message per device lane.
/// This is byte-for-byte equivalent to `write_message_counter` followed by SHA-256.
pub fn hash_message_counter(
    message: &[u8],
    offset: usize,
    length: usize,
    counter: u128,
) -> Result<[u8; 32], WindowError> {
    use sha2::Digest;
    let end = offset
        .checked_add(length)
        .ok_or(WindowError::InvalidBounds)?;
    if length == 0 || end > message.len() {
        return Err(WindowError::InvalidBounds);
    }
    if length < 16 && counter >= (1u128 << (length * 8)) {
        return Err(WindowError::Exhausted);
    }
    let mut hash = Sha256::new();
    hash.update(&message[..offset]);
    let zeros = [0u8; 64];
    let mut padding = length.saturating_sub(16);
    while padding != 0 {
        let chunk = padding.min(zeros.len());
        hash.update(&zeros[..chunk]);
        padding -= chunk;
    }
    let bytes = counter.to_be_bytes();
    hash.update(&bytes[16 - length.min(16)..]);
    hash.update(&message[end..]);
    Ok(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_window_hash_equals_materialized_message() {
        use sha2::Digest;
        for length in [1, 2, 8, 15, 16, 17, 63, 64, 65, 129] {
            for counter in [0, 1, 255] {
                let original = [0xa5; 160];
                let mut materialized = original;
                write_message_counter(&mut materialized, 7, length, counter).unwrap();
                let expected: [u8; 32] = Sha256::digest(materialized).into();
                assert_eq!(
                    hash_message_counter(&original, 7, length, counter).unwrap(),
                    expected
                );
            }
        }
        assert_eq!(
            hash_message_counter(&[], 0, 0, 0),
            Err(WindowError::InvalidBounds)
        );
        assert_eq!(
            hash_message_counter(&[0; 8], usize::MAX, 2, 0),
            Err(WindowError::InvalidBounds)
        );
        assert_eq!(
            hash_message_counter(&[0; 8], 0, 1, 256),
            Err(WindowError::Exhausted)
        );
    }

    #[test]
    fn salt_enumeration_wraps_without_repetition() {
        let mut seen = [false; 256];
        let mut output = [0];
        for counter in 0..256 {
            write_salt_counter(&[173], counter, &mut output).unwrap();
            assert!(!seen[output[0] as usize]);
            seen[output[0] as usize] = true;
        }
        assert_eq!(
            write_salt_counter(&[173], 256, &mut output),
            Err(WindowError::Exhausted)
        );
        let mut wide = [0; 16];
        write_salt_counter(&[255; 16], 1, &mut wide).unwrap();
        assert_eq!(wide, [0; 16]);
        write_salt_counter(&[], 0, &mut []).unwrap();
        assert_eq!(
            write_salt_counter(&[], 1, &mut []),
            Err(WindowError::Exhausted)
        );
    }

    #[test]
    fn window_uniqueness_and_boundaries() {
        let mut message = [0xa5; 6];
        for counter in 0..=255 {
            write_message_counter(&mut message, 2, 1, counter).unwrap();
            assert_eq!(message, [0xa5, 0xa5, counter as u8, 0xa5, 0xa5, 0xa5]);
        }
        let original = message;
        assert_eq!(
            write_message_counter(&mut message, 2, 1, 256),
            Err(WindowError::Exhausted)
        );
        assert_eq!(message, original);
        for (offset, length) in [(0, 0), (6, 1), (usize::MAX, 2), (2, usize::MAX)] {
            assert_eq!(
                write_message_counter(&mut message, offset, length, 0),
                Err(WindowError::InvalidBounds)
            );
            assert_eq!(message, original);
        }
    }

    #[test]
    fn large_window_and_counter_endianness() {
        let mut message = [0xa5; 34];
        write_message_counter(&mut message, 1, 32, 0x0102).unwrap();
        assert_eq!(message[0], 0xa5);
        assert_eq!(message[33], 0xa5);
        assert_eq!(&message[1..31], &[0; 30]);
        assert_eq!(&message[31..33], &[1, 2]);
        write_message_counter(&mut message, 1, 16, u128::MAX).unwrap();
        assert_eq!(&message[1..17], &[255; 16]);
    }

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
