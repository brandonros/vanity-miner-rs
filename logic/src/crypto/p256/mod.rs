//! NIST P-256 primitives. Candidate entropy and output belong to the runners.

use p256::elliptic_curve::sec1::ToEncodedPoint;
use zeroize::Zeroizing;

/// Rejection sampling of uniformly distributed PRF outputs, without modulo bias.
pub fn candidate_scalar(
    deriver: &crate::search::candidate_derivation::CandidateDeriver,
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
pub mod signatures;

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
        use crate::search::candidate_derivation::{CandidateDeriver, CandidateDomain};
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
}
