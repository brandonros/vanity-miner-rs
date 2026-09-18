//! Fixed search inputs and independent verification of completed candidates.
use super::*;
use logic::modes::rsa_modulus::{self as device_logic, Pair, SearchConfig};
use logic::search::candidate_result::BatchResult;
use rand::RngCore;

pub fn run(
    search: &ModulusSearch,
    control: &SearchControl,
    mut evaluate: impl FnMut(&SearchConfig, &HexPattern, u64, u32) -> Result<BatchResult, String>,
) -> Result<(), String> {
    let constraints = search.validate()?;
    let mut seed = Zeroizing::new([0; 32]);
    OsRng
        .try_fill_bytes(seed.as_mut())
        .map_err(|_| "OS cryptographic entropy unavailable")?;
    let config = Zeroizing::new(constraints.device_config(*seed, 0)?);
    let output = crate::runner::batches::search(
        |start, count| evaluate(&config, &constraints.pattern, start, count),
        u64::MAX,
        control,
        |id, bytes| {
            let pair = Zeroizing::new(Pair {
                p: bytes[..128].try_into().unwrap(),
                q: bytes[128..].try_into().unwrap(),
                id,
            });
            verify_pair(&config, &constraints.pattern, &pair).map(Some)
        },
    )?;
    if let Some(output) = output {
        crate::runner::progress::print_verified(control, output)?;
    }
    Ok(())
}

/// The CPU touches factor arithmetic only for a completed candidate pair.
/// Reconstruct p from its PRF input, repeat primality checks with fresh random
/// bases, validate the full key and pattern, then verify a blinded sign/verify.
pub fn verify_pair(
    config: &SearchConfig,
    pattern: &HexPattern,
    pair: &Pair,
) -> Result<String, String> {
    let expected = Zeroizing::new(
        device_logic::generate_p(config, pair.id).ok_or("RSA factor reconstruction failed")?,
    );
    if pair.p != *expected {
        return Err("RSA device p failed reconstruction".into());
    }
    let expected_q = Zeroizing::new(
        device_logic::generate_q(config, &pair.p, pair.id)
            .map_err(|error| format!("RSA candidate range reconstruction failed: {error:?}"))?
            .ok_or("RSA candidate range reconstruction failed")?,
    );
    if pair.q != *expected_q {
        return Err("RSA device q failed reconstruction".into());
    }
    let p = Zeroizing::new(BigUint::from_bytes_be(&pair.p));
    let q = Zeroizing::new(BigUint::from_bytes_be(&pair.q));
    if p.bits() != 1024 || q.bits() != 1024 || !sufficiently_separated(&p, &q) {
        return Err("RSA device pair failed factor bounds or separation".into());
    }
    let mut key = RsaPrivateKey::from_p_q((*p).clone(), (*q).clone(), BigUint::from(65537u32))
        .map_err(|_| "RSA device key construction failed")?;
    validate_rsa2048(&mut key)?;
    if !pattern.matches(fixed_bytes::<256>(key.n())?.as_ref()) {
        return Err("RSA device modulus failed pattern verification".into());
    }
    let digest = Sha256::digest(b"vanity-miner RSA key consistency check");
    let signature = key
        .sign_with_rng(&mut OsRng, Pss::new_blinded::<Sha256>(), &digest)
        .map_err(|_| "RSA consistency signing failed")?;
    key.to_public_key()
        .verify(Pss::new::<Sha256>(), &digest, &signature)
        .map_err(|_| "RSA consistency verification failed")?;
    let private = key
        .to_pkcs8_der()
        .map_err(|_| "RSA private key encoding failed")?;
    Ok(format!(
        "[rsa-modulus] public_key={}\n[rsa-modulus] public_exponent={}\n[rsa-modulus] private_key_pkcs8={}",
        hex::encode(key.n().to_bytes_be()),
        hex::encode(key.e().to_bytes_be()),
        hex::encode(private.as_bytes()),
    ))
}
