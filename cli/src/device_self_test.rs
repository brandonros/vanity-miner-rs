//! Differential fixtures usable with the real CUDA transport via `self-test`.
//! Fresh RSA keys stay in memory. Synthetic P-256 test secrets are public values.
use logic::hex_pattern::HexPattern;
#[cfg(feature = "p256-public-key")]
use logic::p256_public_key_vanity::P256PublicRequest;
#[cfg(feature = "p256-signature")]
use logic::p256_signature_vanity::P256SignatureRequest;
#[cfg(feature = "rsa-modulus")]
use logic::rsa_modulus_vanity::RsaModulusRequest;
#[cfg(feature = "rsa-pss")]
use logic::rsa_pss_signature_vanity::RsaPssRequest;
use zeroize::Zeroizing;

fn compare(
    expected: Zeroizing<Vec<logic::candidate_result::CandidateResult>>,
    actual: Vec<logic::candidate_result::CandidateResult>,
) -> Result<(), String> {
    let actual = Zeroizing::new(actual);
    if actual.len() != expected.len() {
        return Err("crypto differential lane-count mismatch".into());
    }
    for (lane, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        if actual.status != expected.status || actual.bytes != expected.bytes {
            return Err(format!(
                "crypto differential mismatch at lane {lane}; secret bytes suppressed"
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "p256-public-key")]
pub fn p256_public(device: &mut crate::p256_public::EvaluateBatch<'_>) -> Result<(), String> {
    for (target, width) in [(0, 32), (1, 32), (2, 64), (3, 65)] {
        let request = Zeroizing::new(P256PublicRequest {
            seed: [0x42; 32],
            worker: 7,
            target,
            reserved: 0,
        });
        for prefix in ["", "f"] {
            let pattern = HexPattern::new(prefix, "", width).map_err(|e| e.to_string())?;
            compare(
                Zeroizing::new(crate::test_support::p256_public(
                    &request,
                    &pattern,
                    &[],
                    u64::MAX - 7,
                    8,
                )?),
                device(&request, &pattern, &[], u64::MAX - 7, 8)?,
            )?;
        }
    }

    Ok(())
}

#[cfg(feature = "p256-signature")]
pub fn p256_signature(device: &mut crate::p256_signature::EvaluateBatch<'_>) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    let mut private = [0; 32];
    private[31] = 1;
    let public = logic::p256_vanity::public_point(&private).ok_or("test public point failed")?;
    let message = b"header\0\0footer";
    for source in [0, 1] {
        for target in [0, 1, 2] {
            for s_form in [0, 1, 2] {
                let request = Zeroizing::new(P256SignatureRequest {
                    private,
                    seed: [0x42; 32],
                    fingerprint: Sha256::digest(public).into(),
                    digest: Sha256::digest(message).into(),
                    worker: 5,
                    offset: 6,
                    length: 2,
                    source,
                    target,
                    s_form,
                    reserved: 0,
                });
                let width = if target == 0 { 64 } else { 32 };
                let pattern = HexPattern::new("", "", width).map_err(|e| e.to_string())?;
                compare(
                    Zeroizing::new(crate::test_support::p256_signature(
                        &request, &pattern, message, 254, 4,
                    )?),
                    device(&request, &pattern, message, 254, 4)?,
                )?;
            }
        }
    }

    Ok(())
}

#[cfg(feature = "rsa-pss")]
pub fn rsa_pss(device: &mut crate::rsa_pss_search::EvaluateBatch<'_>) -> Result<(), String> {
    use crate::rsa_host::{fixed_bytes, validate_rsa2048};
    use rsa::{RsaPrivateKey, traits::PrivateKeyParts};
    let mut key = RsaPrivateKey::new(&mut rand::rngs::OsRng, 2048)
        .map_err(|_| "self-test RSA key generation failed")?;
    validate_rsa2048(&mut key)?;

    use sha2::{Digest, Sha256};
    let message = b"header\0\0footer";
    let coefficient = Zeroizing::new(
        key.crt_coefficient()
            .ok_or("self-test RSA inverse failed")?,
    );
    for source in [0, 1] {
        for length in [0, 1, 32, 222] {
            let request = Zeroizing::new(RsaPssRequest {
                p: *fixed_bytes(&key.primes()[0])?,
                q: *fixed_bytes(&key.primes()[1])?,
                dp: *fixed_bytes(key.dp().ok_or("self-test RSA dp missing")?)?,
                dq: *fixed_bytes(key.dq().ok_or("self-test RSA dq missing")?)?,
                q_inv: *fixed_bytes(&coefficient)?,
                digest: Sha256::digest(message).into(),
                salt: [255; 222],
                reserved: [0; 2],
                offset: 6,
                length: 2,
                source,
                salt_length: length,
            });
            let pattern = HexPattern::new("", "", 256).map_err(|e| e.to_string())?;
            compare(
                Zeroizing::new(crate::test_support::rsa_pss(
                    &request,
                    &pattern,
                    message,
                    0,
                    if length == 0 && source == 0 { 1 } else { 4 },
                )?),
                device(
                    &request,
                    &pattern,
                    message,
                    0,
                    if length == 0 && source == 0 { 1 } else { 4 },
                )?,
            )?;
        }
    }

    Ok(())
}

#[cfg(feature = "rsa-modulus")]
pub fn rsa_modulus(device: &mut crate::rsa_modulus::EvaluateBatch<'_>) -> Result<(), String> {
    use crate::rsa_host::{fixed_bytes, validate_rsa2048};
    use rsa::{RsaPrivateKey, traits::PrivateKeyParts};
    let mut key = RsaPrivateKey::new(&mut rand::rngs::OsRng, 2048)
        .map_err(|_| "self-test RSA key generation failed")?;
    validate_rsa2048(&mut key)?;

    let q = &key.primes()[1];
    let upper = Zeroizing::new(q + rsa::BigUint::from(8u8));
    let request = Zeroizing::new(RsaModulusRequest {
        p: *fixed_bytes(&key.primes()[0])?,
        first: *fixed_bytes(q)?,
        stride: *fixed_bytes(&rsa::BigUint::from(2u8))?,
        upper: *fixed_bytes(&upper)?,
    });
    let pattern = HexPattern::new("", "", 256).map_err(|e| e.to_string())?;
    // Includes the known prime at index zero and one out-of-range lane.
    compare(
        Zeroizing::new(crate::test_support::rsa_modulus(
            &request,
            &pattern,
            &[],
            0,
            6,
        )?),
        device(&request, &pattern, &[], 0, 6)?,
    )?;

    Ok(())
}
