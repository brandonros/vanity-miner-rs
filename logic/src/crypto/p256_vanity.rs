//! NIST P-256 primitives. Candidate entropy and output belong to the runners.

use p256::elliptic_curve::sec1::ToEncodedPoint;
use zeroize::Zeroizing;

/// Rejection sampling of uniformly distributed PRF outputs, without modulo bias.
pub fn candidate_scalar(
    deriver: &crate::search::crypto_search::CandidateDeriver,
    worker: u64,
    counter: u128,
) -> Option<Zeroizing<[u8; 32]>> {
    for attempt in 0..=u32::MAX {
        let bytes = Zeroizing::new(deriver.block(worker, counter, attempt));
        if p256::SecretKey::from_slice(bytes.as_ref()).is_ok() {
            return Some(bytes);
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PublicTarget {
    X,
    Y,
    Xy,
    Uncompressed,
}

impl PublicTarget {
    pub fn bytes(self, point: &[u8; 65]) -> &[u8] {
        match self {
            Self::X => &point[1..33],
            Self::Y => &point[33..65],
            Self::Xy => &point[1..65],
            Self::Uncompressed => point,
        }
    }

    pub const fn width(self) -> usize {
        match self {
            Self::X | Self::Y => 32,
            Self::Xy => 64,
            Self::Uncompressed => 65,
        }
    }
}

/// Reject zero and out-of-range scalars; never reduce arbitrary bytes modulo n.
pub fn public_point(private: &[u8; 32]) -> Option<[u8; 65]> {
    let key = p256::SecretKey::from_slice(private).ok()?;
    let encoded = key.public_key().to_encoded_point(false);
    encoded.as_bytes().try_into().ok()
}

#[cfg(feature = "p256-signature")]
pub mod signatures {
    use crate::search::hex_pattern::HexPattern;
    use ecdsa::hazmat::SignPrimitive;
    use p256::ecdsa::{
        Signature, SigningKey, VerifyingKey,
        signature::{Signer, Verifier},
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum SignatureTarget {
        Raw,
        R,
        S,
    }

    impl SignatureTarget {
        pub fn bytes(self, signature: &[u8; 64]) -> &[u8] {
            match self {
                Self::Raw => signature,
                Self::R => &signature[..32],
                Self::S => &signature[32..],
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum SForm {
        Low,
        High,
        Either,
    }

    /// RFC 6979 deterministic ECDSA with SHA-256, before S normalization.
    pub fn sign_message(private: &[u8; 32], message: &[u8]) -> Option<[u8; 64]> {
        let key = SigningKey::from_slice(private).ok()?;
        let signature: Signature = key.try_sign(message).ok()?;
        Some(signature.to_bytes().into())
    }

    /// RFC 6979 over an already computed SHA-256 digest (no second hash).
    pub fn sign_digest(private: &[u8; 32], digest: &[u8; 32]) -> Option<[u8; 64]> {
        use p256::ecdsa::signature::hazmat::PrehashSigner;
        let key = SigningKey::from_slice(private).ok()?;
        let signature: Signature = key.sign_prehash(digest).ok()?;
        Some(signature.to_bytes().into())
    }

    /// Sign an SHA-256 digest with an explicitly supplied secret nonce.
    /// The caller must use cryptographic rejection sampling bound to this key
    /// and digest, and must never reuse the nonce for another message.
    pub fn sign_digest_ephemeral(
        private: &[u8; 32],
        digest: &[u8; 32],
        nonce: &[u8; 32],
    ) -> Option<[u8; 64]> {
        let private = p256::SecretKey::from_slice(private).ok()?;
        let nonce = p256::SecretKey::from_slice(nonce).ok()?;
        let d = private.to_nonzero_scalar();
        let k = nonce.to_nonzero_scalar();
        let (signature, _) = d
            .as_ref()
            .try_sign_prehashed(*k.as_ref(), digest.into())
            .ok()?;
        Some(signature.to_bytes().into())
    }

    /// Compute r without calculating s, for direct-ephemeral r-only searches.
    pub fn ephemeral_r(nonce: &[u8; 32]) -> Option<[u8; 32]> {
        use p256::elliptic_curve::{Field, bigint::U256, ops::Reduce};
        let point = super::public_point(nonce)?;
        let x: [u8; 32] = point[1..33].try_into().ok()?;
        let r = <p256::Scalar as Reduce<U256>>::reduce_bytes(&x.into());
        if bool::from(r.is_zero()) {
            return None;
        }
        Some(r.to_bytes().into())
    }

    /// Match r before performing the private-key-dependent signature operation.
    pub fn matching_ephemeral_signature(
        private: &[u8; 32],
        digest: &[u8; 32],
        nonce: &[u8; 32],
        target: SignatureTarget,
        form: SForm,
        pattern: &HexPattern,
    ) -> Option<[u8; 64]> {
        if target == SignatureTarget::R && !pattern.matches(&ephemeral_r(nonce)?) {
            return None;
        }
        let raw = sign_digest_ephemeral(private, digest, nonce)?;
        matching_representation(&raw, target, form, pattern)
    }

    /// Match precisely the representation returned to the caller for emission.
    pub fn matching_representation(
        raw: &[u8; 64],
        target: SignatureTarget,
        form: SForm,
        pattern: &HexPattern,
    ) -> Option<[u8; 64]> {
        let signature = Signature::from_slice(raw).ok()?;
        let low = signature.normalize_s().unwrap_or(signature);
        let (r, s) = low.split_scalars();
        let high = Signature::from_scalars(r.to_bytes(), (-s).to_bytes()).ok()?;
        let choices = match form {
            SForm::Low => [low, low],
            SForm::High => [high, high],
            SForm::Either => [low, high],
        };
        for choice in choices {
            let bytes = choice.to_bytes().into();
            if pattern.matches(target.bytes(&bytes)) {
                return Some(bytes);
            }
        }
        None
    }

    pub fn verify(public: &[u8; 65], message: &[u8], raw: &[u8; 64]) -> bool {
        let Ok(key) = VerifyingKey::from_sec1_bytes(public) else {
            return false;
        };
        let Ok(signature) = Signature::from_slice(raw) else {
            return false;
        };
        key.verify(message, &signature).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scalar() -> [u8; 32] {
        // Public test value only; never used by a candidate generator.
        let mut scalar = [0; 32];
        scalar[31] = 1;
        scalar
    }

    #[test]
    fn generator_and_target_encodings() {
        let point = public_point(&test_scalar()).unwrap();
        assert_eq!(
            hex::encode(point),
            concat!(
                "04",
                "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296",
                "4fe342e2fe1a7f9b8ee7eb4a7c0f9e162bce33576b315ececbb6406837bf51f5"
            )
        );
        for target in [
            PublicTarget::X,
            PublicTarget::Y,
            PublicTarget::Xy,
            PublicTarget::Uncompressed,
        ] {
            assert_eq!(target.bytes(&point).len(), target.width());
        }
        assert_eq!(&point[1..33], PublicTarget::X.bytes(&point));
        assert_eq!(&point[33..], PublicTarget::Y.bytes(&point));
    }

    #[test]
    fn invalid_scalars() {
        assert!(public_point(&[0; 32]).is_none());
        assert!(public_point(&[255; 32]).is_none());
        let order = hex::decode("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551")
            .unwrap();
        assert!(public_point(&order.try_into().unwrap()).is_none());
    }

    #[test]
    fn derived_scalars_are_valid_and_reproducible() {
        use crate::search::crypto_search::{CandidateDeriver, CandidateDomain};
        let deriver = CandidateDeriver::new(
            [0x42; 32],
            CandidateDomain::P256PrivateKey,
            [0; 32],
            [0; 32],
        );
        for worker in 0..4 {
            for counter in 0..8 {
                let scalar = candidate_scalar(&deriver, worker, counter).unwrap();
                assert!(public_point(&scalar).is_some());
                // Avoid assertion diagnostics that would print candidate scalars.
                assert!(
                    scalar.as_ref()
                        == candidate_scalar(&deriver, worker, counter)
                            .unwrap()
                            .as_ref()
                );
            }
        }
    }

    #[cfg(feature = "p256-signature")]
    #[test]
    fn ephemeral_signatures_verify_and_reject_invalid_nonces() {
        use crate::crypto::sha256::Sha256;
        use crate::search::crypto_search::{CandidateDeriver, CandidateDomain};
        let private = test_scalar();
        let public = public_point(&private).unwrap();
        let message = b"ephemeral signing public test";
        let digest: [u8; 32] = Sha256::digest(message);
        let deriver = CandidateDeriver::new(
            [0x42; 32],
            CandidateDomain::P256Ephemeral,
            Sha256::digest(public),
            digest,
        );
        for counter in 0..4 {
            let nonce = candidate_scalar(&deriver, 0, counter).unwrap();
            let raw = signatures::sign_digest_ephemeral(&private, &digest, &nonce).unwrap();
            assert!(signatures::verify(&public, message, &raw));
            let r = signatures::ephemeral_r(&nonce).unwrap();
            assert_eq!(&raw[..32], &r);
            let pattern =
                crate::search::hex_pattern::HexPattern::new(&hex::encode(r), "", 32).unwrap();
            let matched = signatures::matching_ephemeral_signature(
                &private,
                &digest,
                &nonce,
                signatures::SignatureTarget::R,
                signatures::SForm::Low,
                &pattern,
            )
            .unwrap();
            assert!(signatures::verify(&public, message, &matched));
        }
        assert!(signatures::sign_digest_ephemeral(&private, &digest, &[0; 32]).is_none());
        assert!(signatures::sign_digest_ephemeral(&private, &digest, &[255; 32]).is_none());
    }

    #[cfg(feature = "p256-signature")]
    #[test]
    fn message_window_signatures_verify() {
        let private = test_scalar();
        let public = public_point(&private).unwrap();
        let original = b"header\0\0\0\0footer";
        let mut message = *original;
        let mut previous = None;
        for counter in 0..4 {
            crate::search::crypto_search::write_message_counter(&mut message, 6, 4, counter)
                .unwrap();
            assert_eq!(&message[..6], &original[..6]);
            assert_eq!(&message[10..], &original[10..]);
            let signature = signatures::sign_message(&private, &message).unwrap();
            assert!(signatures::verify(&public, &message, &signature));
            assert_ne!(previous, Some(signature));
            previous = Some(signature);
        }
    }

    #[cfg(feature = "p256-signature")]
    #[test]
    fn rfc6979_known_answers() {
        // Published RFC 6979 A.2.5 test key, not a generated or real user key.
        let private: [u8; 32] =
            hex::decode("c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721")
                .unwrap()
                .try_into()
                .unwrap();
        for (message, expected) in [
            (
                b"sample".as_slice(),
                concat!(
                    "efd48b2aacb6a8fd1140dd9cd45e81d69d2c877b56aaf991c34d0ea84eaf3716",
                    "f7cb1c942d657c41d436c7a1b6e29f65f3e900dbb9aff4064dc4ab2f843acda8"
                ),
            ),
            (
                b"test".as_slice(),
                concat!(
                    "f1abb023518351cd71d881567b1ea663ed3efcf6c5132b354f28d3b0b7d38367",
                    "019f4113742a2b14bd25926b49c649155f267e60d3814b4c0cc84250e46f0083"
                ),
            ),
        ] {
            let raw = signatures::sign_message(&private, message).unwrap();
            assert_eq!(hex::encode(raw), expected);
            let signature = p256::ecdsa::Signature::from_slice(&raw).unwrap();
            let der = signature.to_der();
            assert_eq!(
                p256::ecdsa::Signature::from_der(der.as_bytes()).unwrap(),
                signature
            );
        }
    }

    #[cfg(feature = "p256-signature")]
    #[test]
    fn signatures_and_exact_s_matching() {
        use crate::search::hex_pattern::HexPattern;
        use signatures::*;
        let private = test_scalar();
        let public = public_point(&private).unwrap();
        let raw = sign_message(&private, b"public test message").unwrap();
        assert_eq!(Some(raw), sign_message(&private, b"public test message"));
        assert!(verify(&public, b"public test message", &raw));
        assert!(!verify(&public, b"different message", &raw));
        let all = HexPattern::new("", "", 64).unwrap();
        let low = matching_representation(&raw, SignatureTarget::Raw, SForm::Low, &all).unwrap();
        let high = matching_representation(&raw, SignatureTarget::Raw, SForm::High, &all).unwrap();
        assert_eq!(&low[..32], &high[..32]);
        assert_ne!(&low[32..], &high[32..]);
        assert!(verify(&public, b"public test message", &low));
        assert!(verify(&public, b"public test message", &high));
        for target in [SignatureTarget::Raw, SignatureTarget::R, SignatureTarget::S] {
            let text = hex::encode(target.bytes(&high));
            let pattern = HexPattern::new(&text, "", text.len() / 2).unwrap();
            let chosen = matching_representation(&raw, target, SForm::Either, &pattern).unwrap();
            assert!(pattern.matches(target.bytes(&chosen)));
            assert!(verify(&public, b"public test message", &chosen));
        }
    }
}
