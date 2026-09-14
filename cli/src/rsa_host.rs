//! RSA host validation used before key export or device setup.

use num_bigint_dig::{BigUint, RandBigInt, prime::probably_prime};
use rand::rngs::OsRng;
use rsa::{
    RsaPrivateKey,
    traits::{PrivateKeyParts, PublicKeyParts},
};
use zeroize::Zeroizing;

pub fn fixed_bytes<const N: usize>(value: &BigUint) -> Result<Zeroizing<[u8; N]>, String> {
    let bytes = Zeroizing::new(value.to_bytes_be());
    if bytes.len() > N {
        return Err("RSA component exceeds the required width".into());
    }
    let mut output = Zeroizing::new([0; N]);
    output[N - bytes.len()..].copy_from_slice(&bytes);
    Ok(output)
}

/// The library's Baillie-PSW/Miller-Rabin filter is supplemented with 64 fresh
/// uniform OS-random bases, giving a <=2^-128 Miller-Rabin bound even for a
/// composite selected by an adversary before the independent bases are drawn.
pub fn strong_probable_prime(n: &BigUint) -> bool {
    if !probably_prime(n, 32) {
        return false;
    }
    if n <= &BigUint::from(3u8) {
        return true;
    }
    let one = BigUint::from(1u8);
    let two = BigUint::from(2u8);
    let minus_one = Zeroizing::new(n - &one);
    let shifts = minus_one.trailing_zeros().unwrap_or(0);
    let odd = Zeroizing::new(&*minus_one >> shifts);
    'round: for _ in 0..64 {
        let base = Zeroizing::new(OsRng.gen_biguint_range(&two, &minus_one));
        let mut y = Zeroizing::new(base.modpow(&odd, n));
        if *y == one || *y == *minus_one {
            continue;
        }
        for _ in 1..shifts {
            *y = (&*y * &*y) % n;
            if *y == *minus_one {
                continue 'round;
            }
            if *y == one {
                return false;
            }
        }
        return false;
    }
    true
}

pub fn sufficiently_separated(p: &BigUint, q: &BigUint) -> bool {
    let distance = Zeroizing::new(if p >= q { p - q } else { q - p });
    *distance > (BigUint::from(1u8) << 924usize)
}

pub fn validate_rsa2048(key: &mut RsaPrivateKey) -> Result<(), String> {
    if key.n().bits() != 2048
        || key.e() != &BigUint::from(65537u32)
        || key.primes().len() != 2
        || key.primes().iter().any(|p| p.bits() != 1024)
    {
        return Err(
            "RSA modes require a two-prime RSA-2048 key, 1024-bit factors, and exponent 65537"
                .into(),
        );
    }
    if !sufficiently_separated(&key.primes()[0], &key.primes()[1]) {
        return Err("RSA prime factors do not meet the minimum separation".into());
    }
    key.validate()
        .map_err(|_| "RSA component relationships are invalid")?;
    if !key.primes().iter().all(strong_probable_prime) {
        return Err("RSA factor failed independent probable-prime validation".into());
    }
    key.precompute()
        .map_err(|_| "RSA CRT precomputation failed")?;
    Ok(())
}

#[cfg(feature = "rsa-pss")]
pub fn crt_for_key(key: &RsaPrivateKey) -> Result<logic::rsa_crt::Rsa2048Crt, String> {
    let p = fixed_bytes(&key.primes()[0])?;
    let q = fixed_bytes(&key.primes()[1])?;
    let dp = fixed_bytes(key.dp().ok_or("missing RSA dp")?)?;
    let dq = fixed_bytes(key.dq().ok_or("missing RSA dq")?)?;
    let coefficient = Zeroizing::new(key.crt_coefficient().ok_or("invalid RSA CRT inverse")?);
    let q_inv = fixed_bytes(&coefficient)?;
    logic::rsa_crt::Rsa2048Crt::new(&p, &q, &dp, &dq, &q_inv)
        .ok_or_else(|| "RSA fixed-width CRT setup failed".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probable_prime_and_factor_distance() {
        for n in [0u64, 1, 4, 9, 15, 341, 561, 1105, 3215031751] {
            assert!(!strong_probable_prime(&BigUint::from(n)));
        }
        for n in [2u64, 3, 5, 61, 65537] {
            assert!(strong_probable_prime(&BigUint::from(n)));
        }
        let p = BigUint::from(1u8) << 1023usize;
        assert!(!sufficiently_separated(
            &p,
            &(&p + (BigUint::from(1u8) << 924usize))
        ));
        assert!(sufficiently_separated(
            &p,
            &(&p + (BigUint::from(1u8) << 925usize))
        ));
    }
}
