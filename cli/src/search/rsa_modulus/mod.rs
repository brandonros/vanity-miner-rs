//! Constructive RSA-2048 modulus search: interval and residue constraints on q.

use crate::{
    rsa_host::{fixed_bytes, sufficiently_separated, validate_rsa2048},
    search_control::SearchControl,
};
use logic::crypto::sha256::Sha256;
use logic::search::hex_pattern::HexPattern;
use num_bigint_dig::{BigUint, ModInverse, RandBigInt, prime::probably_prime};
use rand::rngs::OsRng;
use rsa::{Pss, RsaPrivateKey, pkcs8::EncodePrivateKey, traits::PublicKeyParts};
use std::{sync::Arc, thread, time::Duration};
use zeroize::Zeroizing;

pub struct ModulusSearch {
    pub prefix: String,
    pub suffix: String,
    pub workers: usize,
}

pub struct ModulusReport {
    pub q_candidates_tested: u64,
    pub elapsed: Duration,
    pub found: bool,
    pub output: Option<String>,
}

pub struct ModulusConstraints {
    pub pattern: HexPattern,
    lower: BigUint,
    upper: BigUint,
    suffix: BigUint,
    suffix_bits: usize,
}

/// Secret progression parameters must not be logged or serialized unprotected.
pub struct QProgression {
    pub first: Zeroizing<BigUint>,
    pub stride: BigUint,
    pub count: Zeroizing<BigUint>,
}

impl QProgression {
    fn search_budget(&self) -> u64 {
        if *self.count >= BigUint::from(65536u32) {
            65536
        } else {
            self.count
                .to_bytes_be()
                .iter()
                .fold(0u64, |n, byte| (n << 8) | u64::from(*byte))
        }
    }
}

fn ceil_div(n: &BigUint, d: &BigUint) -> BigUint {
    (n + d - BigUint::from(1u8)) / d
}

impl ModulusConstraints {
    pub fn new(prefix: &str, suffix: &str) -> Result<Self, String> {
        let mut pattern = HexPattern::new(prefix, suffix, 256).map_err(|e| e.to_string())?;
        pattern
            .constrain_byte(0, 128, 128)
            .map_err(|e| e.to_string())?;
        pattern
            .constrain_byte(255, 1, 1)
            .map_err(|e| e.to_string())?;
        // Bound the requested pattern, while permitting tiny q intervals for
        // experimental long-prefix searches. The device stride is 1024 bits.
        if prefix.len() + suffix.len() > 256 || suffix.len() >= 256 {
            return Err("RSA prefix and suffix may constrain at most 128 bytes combined; suffix alone must be shorter than 128 bytes".into());
        }
        let one = BigUint::from(1u8);
        let (lower, upper) = if prefix.is_empty() {
            (&one << 2047usize, (&one << 2048usize) - &one)
        } else {
            let p = BigUint::parse_bytes(prefix.as_bytes(), 16).ok_or("invalid prefix")?;
            let shift = 2048 - prefix.len() * 4;
            (&p << shift, ((p + &one) << shift) - &one)
        };
        let suffix_bits = (suffix.len() * 4).max(1);
        let suffix = if suffix.is_empty() {
            one.clone()
        } else {
            BigUint::parse_bytes(suffix.as_bytes(), 16).ok_or("invalid suffix")?
        };
        // Both factors are odd, at most M = 2^1024 - 1, and must differ
        // by more than 2^924. Their minimum allowed (even) difference is
        // 2^924 + 2, so no accepted modulus can exceed M * (M - 2^924 - 2).
        // This is a necessary feasibility check, not a guarantee of primes.
        let max_factor = (&one << 1024usize) - &one;
        let max_modulus = &max_factor * (&max_factor - (&one << 924usize) - BigUint::from(2u8));
        // Find the smallest value in the prefix interval that also has the
        // requested suffix (or odd parity when the suffix is unconstrained).
        let stride = &one << suffix_bits;
        let first = &lower + ((&suffix + &stride - (&lower % &stride)) % &stride);
        if first > upper || first > max_modulus {
            return Err("RSA prefix/suffix cannot satisfy the required factor separation |p - q| > 2^924; shorten or change the pattern".into());
        }
        Ok(Self {
            pattern,
            lower,
            upper,
            suffix,
            suffix_bits,
        })
    }

    /// Select uniformly from feasible odd 1024-bit p values. Conditioning p on
    /// a nonempty q interval avoids expensive retries for prefixes near ff... .
    fn random_p_candidate(&self) -> BigUint {
        let one = BigUint::from(1u8);
        let max = (&one << 1024usize) - &one;
        let mut min = ceil_div(&self.lower, &max).max(&one << 1023usize);
        if &min % 2u8 == BigUint::from(0u8) {
            min += &one;
        }
        let count = (&max - &min) / 2u8 + &one;
        min + OsRng.gen_biguint_below(&count) * 2u8
    }

    /// q in [ceil(L/p), floor(U/p)] intersected with the 1024-bit range,
    /// restricted to q = suffix * p^-1 (mod 2^suffix_bits).
    pub fn progression(&self, p: &BigUint) -> Option<QProgression> {
        if p.bits() != 1024 || p % 2u8 == BigUint::from(0u8) {
            return None;
        }
        let one = BigUint::from(1u8);
        let min = Zeroizing::new(ceil_div(&self.lower, p).max(&one << 1023usize));
        let max = Zeroizing::new((&self.upper / p).min((&one << 1024usize) - &one));
        if *min > *max {
            return None;
        }
        let stride = &one << self.suffix_bits;
        let inverse = Zeroizing::new(p.mod_inverse(&stride)?.to_biguint()?);
        let residue = Zeroizing::new((&self.suffix * &*inverse) % &stride);
        let first = Zeroizing::new(&*min + ((&*residue + &stride - (&*min % &stride)) % &stride));
        if *first > *max {
            return None;
        }
        let count = Zeroizing::new((&*max - &*first) / &stride + &one);
        Some(QProgression {
            first,
            stride,
            count,
        })
    }
}

impl ModulusSearch {
    pub fn validate(&self) -> Result<ModulusConstraints, String> {
        let constraints = ModulusConstraints::new(&self.prefix, &self.suffix)?;
        if self.workers == 0 {
            return Err("worker count must be nonzero".into());
        }
        Ok(constraints)
    }
}

fn construct_worker(
    constraints: &ModulusConstraints,
    control: &SearchControl,
) -> Result<Option<RsaPrivateKey>, String> {
    let _stop_peers = control.cancel_on_exit();
    let zero = BigUint::from(0u8);
    let one = BigUint::from(1u8);
    let e = BigUint::from(65537u32);
    while !control.stopped() {
        let p = Zeroizing::new(constraints.random_p_candidate());
        if (&*p - &one) % &e == zero {
            continue;
        }
        let Some(progression) = constraints.progression(&p) else {
            continue;
        };
        if !probably_prime(&p, 32) {
            continue;
        }
        let start = Zeroizing::new(OsRng.gen_biguint_below(&progression.count));
        // Each p gets a securely randomized start and a nonrepeating progression.
        // Exhaust a small interval once, without retesting the same q.
        for counter in 0..progression.search_budget() {
            if control.stopped() {
                return Ok(None);
            }
            let index = Zeroizing::new((&*start + BigUint::from(counter)) % &*progression.count);
            let q = Zeroizing::new(&*progression.first + &*index * &progression.stride);
            control.add_tested(1);
            if (&*q - &one) % &e == zero
                || !sufficiently_separated(&p, &q)
                || !probably_prime(&q, 32)
            {
                continue;
            }
            let mut key = RsaPrivateKey::from_p_q((*p).clone(), (*q).clone(), e.clone())
                .map_err(|_| "RSA key construction failed")?;
            validate_rsa2048(&mut key)?;
            let modulus = fixed_bytes::<256>(key.n())?;
            if !constraints.pattern.matches(modulus.as_ref()) {
                return Err("constructed RSA modulus failed the pattern".into());
            }
            // Independently verify a blinded private/public operation before export.
            let digest = Sha256::digest(b"vanity-miner RSA key consistency check");
            let signature = key
                .sign_with_rng(&mut OsRng, Pss::new_blinded::<Sha256>(), &digest)
                .map_err(|_| "RSA consistency signing failed")?;
            key.to_public_key()
                .verify(Pss::new::<Sha256>(), &digest, &signature)
                .map_err(|_| "RSA consistency verification failed")?;
            if control.claim_verified_winner() {
                return Ok(Some(key));
            }
            return Ok(None);
        }
    }
    Ok(None)
}

fn construct_device(
    constraints: &ModulusConstraints,
    control: &SearchControl,
    device: &mut EvaluateBatch<'_>,
) -> Result<Option<RsaPrivateKey>, String> {
    use logic::modes::rsa_modulus_vanity::RsaModulusRequest;
    let _stop = control.cancel_on_exit();
    let one = BigUint::from(1u8);
    let e = BigUint::from(65537u32);
    while !control.stopped() {
        // Filter independent p candidates on the device. The CPU only samples
        // an eligible odd interval and independently validates the returned p.
        let max = (&one << 1024usize) - &one;
        let mut min = ceil_div(&constraints.lower, &max).max(&one << 1023usize);
        if &min % 2u8 == BigUint::from(0u8) {
            min += &one;
        }
        let candidates = (&max - &min) / 2u8 + &one;
        let p_count = if candidates >= BigUint::from(control.batch_size()) {
            control.batch_size()
        } else {
            candidates
                .to_bytes_be()
                .iter()
                .fold(0u32, |n, b| (n << 8) | u32::from(*b))
        };
        let offset =
            Zeroizing::new(OsRng.gen_biguint_below(&(&candidates - BigUint::from(p_count - 1))));
        let first_p = Zeroizing::new(&min + &*offset * 2u8);
        let upper_p = Zeroizing::new(&*first_p + BigUint::from(p_count - 1) * 2u8);
        let p_request = Zeroizing::new(RsaModulusRequest {
            stage: 1,
            reserved: 0,
            p: [0; 128],
            first: *fixed_bytes(&first_p)?,
            stride: *fixed_bytes(&BigUint::from(2u8))?,
            upper: *fixed_bytes(&upper_p)?,
        });
        if !control.reserve_device_launch() {
            return Ok(None);
        }
        let results = Zeroizing::new(device(&p_request, &constraints.pattern, &[], 0, p_count)?);
        let winner = results.winner(p_count)?;
        control.add_tested(u64::from(p_count));
        if control.stopped() {
            return Ok(None);
        }
        let Some((lane, result)) = winner else {
            continue;
        };
        let p = Zeroizing::new(&*first_p + BigUint::from(lane) * 2u8);
        if result.bytes[..128] != fixed_bytes::<128>(&p)?[..] {
            return Err("device RSA p factor failed reconstruction".into());
        }
        if (&*p - &one) % &e == BigUint::from(0u8) || !crate::rsa_host::strong_probable_prime(&p) {
            continue;
        }
        let Some(progression) = constraints.progression(&p) else {
            continue;
        };
        let budget = progression.search_budget();
        // Choose a start with room for the selected interval, even for one q.
        let start = Zeroizing::new(
            OsRng.gen_biguint_below(&(&*progression.count - BigUint::from(budget - 1))),
        );
        let first = Zeroizing::new(&*progression.first + &*start * &progression.stride);
        let upper = Zeroizing::new(&*first + BigUint::from(budget - 1) * &progression.stride);
        let request = Zeroizing::new(RsaModulusRequest {
            stage: 0,
            reserved: 0,
            p: *fixed_bytes(&p)?,
            first: *fixed_bytes(&first)?,
            stride: *fixed_bytes(&progression.stride)?,
            upper: *fixed_bytes(&upper)?,
        });
        for start in (0..budget).step_by(control.batch_size() as usize) {
            if control.stopped() {
                return Ok(None);
            }
            let count = (budget - start).min(u64::from(control.batch_size())) as u32;
            if !control.reserve_device_launch() {
                return Ok(None);
            }
            let results =
                Zeroizing::new(device(&request, &constraints.pattern, &[], start, count)?);
            let winner = results.winner(count)?;
            control.add_tested(u64::from(count));
            if let Some((lane, result)) = winner {
                if control.stopped() {
                    return Ok(None);
                }
                let q = Zeroizing::new(
                    &*first + BigUint::from(start + lane as u64) * &progression.stride,
                );
                if result.bytes[..128] != fixed_bytes::<128>(&q)?[..] {
                    return Err("device RSA factor failed reconstruction".into());
                }
                if !sufficiently_separated(&p, &q) || !crate::rsa_host::strong_probable_prime(&q) {
                    continue;
                }
                let mut key = RsaPrivateKey::from_p_q((*p).clone(), (*q).clone(), e.clone())
                    .map_err(|_| "device RSA key construction failed")?;
                validate_rsa2048(&mut key)?;
                if !constraints
                    .pattern
                    .matches(&fixed_bytes::<256>(key.n())?[..])
                {
                    return Err("device RSA modulus failed pattern verification".into());
                }
                let digest = Sha256::digest(b"vanity-miner RSA key consistency check");
                let signature = key
                    .sign_with_rng(&mut OsRng, Pss::new_blinded::<Sha256>(), &digest)
                    .map_err(|_| "device RSA consistency signing failed")?;
                key.to_public_key()
                    .verify(Pss::new::<Sha256>(), &digest, &signature)
                    .map_err(|_| "device RSA consistency verification failed")?;
                return Ok(control.claim_verified_winner().then_some(key));
            }
        }
    }
    Ok(None)
}

pub fn run_cpu(
    config: &ModulusSearch,
    control: Arc<SearchControl>,
) -> Result<ModulusReport, String> {
    run(config, control, None)
}

pub fn run_device(
    config: &ModulusSearch,
    control: Arc<SearchControl>,
    device: &mut EvaluateBatch<'_>,
) -> Result<ModulusReport, String> {
    run(config, control, Some(device))
}

fn run(
    config: &ModulusSearch,
    control: Arc<SearchControl>,
    device: Option<&mut EvaluateBatch<'_>>,
) -> Result<ModulusReport, String> {
    let constraints = config.validate()?;
    let outcome = if let Some(device) = device {
        construct_device(&constraints, &control, device)?
    } else {
        thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..config.workers {
                handles.push(scope.spawn(|| construct_worker(&constraints, &control)));
            }
            super::worker_results::join(handles, &control, "RSA modulus worker panicked")
        })?
    };
    let found = outcome.is_some();
    let mut output = None;
    if let Some(key) = outcome {
        let private = key
            .to_pkcs8_der()
            .map_err(|_| "RSA private key encoding failed")?;
        output = Some(format!(
            "[rsa-modulus] public_key={}\n[rsa-modulus] public_exponent={}\n[rsa-modulus] private_key_pkcs8={}",
            hex::encode(key.n().to_bytes_be()),
            hex::encode(key.e().to_bytes_be()),
            hex::encode(private.as_bytes()),
        ));
    }
    if let Some(record) = &output {
        println!("{record}");
    }
    let (q_candidates_tested, elapsed) = control.statistics();
    Ok(ModulusReport {
        q_candidates_tested,
        elapsed,
        found,
        output,
    })
}

pub type EvaluateBatch<'a> = dyn FnMut(
        &logic::modes::rsa_modulus_vanity::RsaModulusRequest,
        &logic::search::hex_pattern::HexPattern,
        &[u8],
        u64,
        u32,
    ) -> Result<logic::search::candidate_result::BatchResult, String>
    + 'a;

#[cfg(test)]
mod tests;
