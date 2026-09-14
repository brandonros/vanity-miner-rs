//! Constructive RSA-2048 modulus search: interval and residue constraints on q.

use crate::{
    protected_output::{StagedOutput, validate_outputs},
    rsa_host::{fixed_bytes, sufficiently_separated, validate_rsa2048},
    search_control::SearchControl,
};
use logic::hex_pattern::HexPattern;
use num_bigint_dig::{BigUint, ModInverse, RandBigInt, prime::probably_prime};
use rand::rngs::OsRng;
use rsa::{
    Pss, RsaPrivateKey,
    pkcs8::{EncodePrivateKey, EncodePublicKey, LineEnding},
    traits::PublicKeyParts,
};
use sha2::{Digest, Sha256};
use std::{path::PathBuf, sync::Arc, thread, time::Duration};
use zeroize::Zeroizing;

pub struct ModulusSearch {
    pub prefix: String,
    pub suffix: String,
    pub private_out: PathBuf,
    pub public_out: PathBuf,
    pub force: bool,
    pub workers: usize,
}

pub struct ModulusReport {
    pub q_candidates_tested: u64,
    pub elapsed: Duration,
    pub found: bool,
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
        // Reserve at least 256 bits of q search entropy; actual intervals are
        // checked as well because integer division and range intersection narrow them.
        if (prefix.len() * 4).max(1) + (suffix.len() * 4).max(1) > 767 {
            return Err("RSA constraints leave insufficient room for 256 bits of secret-factor search entropy".into());
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
            one
        } else {
            BigUint::parse_bytes(suffix.as_bytes(), 16).ok_or("invalid suffix")?
        };
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
        if *count < (&one << 256usize) {
            return None;
        }
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
        validate_outputs(&[&self.private_out, &self.public_out], &[], self.force)
            .map_err(|e| e.to_string())?;
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
        // The >=2^256 progression is far larger than this per-p search budget.
        for counter in 0..65536u32 {
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
    use logic::rsa_modulus_vanity::RsaModulusRequest;
    let _stop = control.cancel_on_exit();
    let one = BigUint::from(1u8);
    let e = BigUint::from(65537u32);
    while !control.stopped() {
        let p = Zeroizing::new(constraints.random_p_candidate());
        if (&*p - &one) % &e == BigUint::from(0u8) || !probably_prime(&p, 32) {
            continue;
        }
        let Some(progression) = constraints.progression(&p) else {
            continue;
        };
        // Choose a start with room for the entire unique 65536-candidate batch.
        let start = Zeroizing::new(
            OsRng.gen_biguint_below(&(&*progression.count - BigUint::from(65535u32))),
        );
        let first = Zeroizing::new(&*progression.first + &*start * &progression.stride);
        let upper = Zeroizing::new(&*first + BigUint::from(65535u32) * &progression.stride);
        let request = Zeroizing::new(RsaModulusRequest {
            p: *fixed_bytes(&p)?,
            first: *fixed_bytes(&first)?,
            stride: *fixed_bytes(&progression.stride)?,
            upper: *fixed_bytes(&upper)?,
        });
        for start in (0..65536u64).step_by(64) {
            if control.stopped() {
                return Ok(None);
            }
            let results = Zeroizing::new(device(
                &request,
                &constraints.pattern,
                &[],
                start,
                64,
            )?);
            if results.len() != 64 {
                return Err("device returned incorrect RSA lane count".into());
            }
            control.add_tested(64);
            for (lane, result) in results.iter().enumerate() {
                if control.stopped() {
                    return Ok(None);
                }
                if result.status == 0 {
                    continue;
                }
                if result.status != 1 {
                    return Err("device RSA modulus evaluation failed".into());
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
            let mut winner = None;
            let mut error = None;
            for handle in handles {
                match handle.join() {
                    Ok(Ok(Some(key))) => winner = Some(key),
                    Ok(Ok(None)) => {}
                    Ok(Err(message)) => {
                        control.cancel();
                        error = Some(message);
                    }
                    Err(_) => {
                        control.cancel();
                        error = Some("RSA modulus worker panicked".into());
                    }
                }
            }
            if let Some(error) = error {
                Err(error)
            } else {
                Ok(winner)
            }
        })?
    };
    let found = outcome.is_some();
    if let Some(key) = outcome {
        let private = key
            .to_pkcs8_pem(LineEnding::LF)
            .map_err(|_| "RSA private PEM encoding failed")?;
        let public = key
            .to_public_key()
            .to_public_key_pem(LineEnding::LF)
            .map_err(|_| "RSA public PEM encoding failed")?;
        let private_stage =
            StagedOutput::new(&config.private_out, private.as_bytes(), config.force)
                .map_err(|e| e.to_string())?;
        let public_stage = StagedOutput::new(&config.public_out, public.as_bytes(), config.force)
            .map_err(|e| e.to_string())?;
        public_stage.publish().map_err(|e| e.to_string())?;
        private_stage.publish().map_err(|e| {
            format!("private output publication failed; public output may exist: {e}")
        })?;
    }
    let (q_candidates_tested, elapsed) = control.statistics();
    Ok(ModulusReport {
        q_candidates_tested,
        elapsed,
        found,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;
    use rsa::{
        pkcs8::{DecodePrivateKey, DecodePublicKey},
        traits::PrivateKeyParts,
    };

    struct Directory(PathBuf);
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn constructive_keys_round_trip_and_pass_libressl_checks() {
        check_runner(false);
        check_runner(true);
    }

    fn check_runner(device: bool) {
        let mut random = [0; 16];
        OsRng.fill_bytes(&mut random);
        let dir = Directory(
            std::env::temp_dir().join(format!("vanity-modulus-test-{}", hex::encode(random))),
        );
        std::fs::create_dir(&dir.0).unwrap();
        for (i, (prefix, suffix)) in [("abc", ""), ("", "fed"), ("d", "b")]
            .into_iter()
            .enumerate()
        {
            let config = ModulusSearch {
                prefix: prefix.into(),
                suffix: suffix.into(),
                private_out: dir.0.join(format!("private-{i}.pem")),
                public_out: dir.0.join(format!("public-{i}.pem")),
                force: false,
                workers: 2,
            };
            let control = Arc::new(SearchControl::new());
            let report = if device {
                run_device(&config, control, &mut crate::test_support::rsa_modulus)
            } else {
                run_cpu(&config, control)
            }
            .unwrap();
            assert!(report.found);
            assert!(report.q_candidates_tested > 0);
            let pem = Zeroizing::new(std::fs::read_to_string(&config.private_out).unwrap());
            let key = RsaPrivateKey::from_pkcs8_pem(&pem).unwrap();
            key.validate().unwrap();
            assert_eq!(key.n().bits(), 2048);
            assert!(key.primes().iter().all(|prime| prime.bits() == 1024));
            assert!(sufficiently_separated(&key.primes()[0], &key.primes()[1]));
            assert!(config.validate().is_err()); // Existing private output is protected.
            let constraints = ModulusConstraints::new(prefix, suffix).unwrap();
            assert!(constraints.pattern.matches(&key.n().to_bytes_be()));
            let public_pem = std::fs::read_to_string(&config.public_out).unwrap();
            let public = rsa::RsaPublicKey::from_public_key_pem(&public_pem).unwrap();
            assert!(public == key.to_public_key());
            if std::process::Command::new("openssl")
                .arg("version")
                .output()
                .is_ok()
            {
                let result = std::process::Command::new("openssl")
                    .args(["rsa", "-in"])
                    .arg(&config.private_out)
                    .args(["-check", "-noout"])
                    .output()
                    .unwrap();
                assert!(result.status.success());
            }
        }
    }

    #[test]
    fn impossible_patterns_and_entropy_constraints() {
        for (prefix, suffix) in [("7", ""), ("", "e"), ("0x8", ""), ("g", "")] {
            assert!(ModulusConstraints::new(prefix, suffix).is_err());
        }
        assert!(ModulusConstraints::new(&"f".repeat(192), "").is_err());
    }

    #[test]
    fn interval_and_residue_match_all_sampled_candidates() {
        for (prefix, suffix) in [("abc", ""), ("", "fed"), ("d", "b"), ("fffffff", "1")] {
            let constraints = ModulusConstraints::new(prefix, suffix).unwrap();
            let (p, progression) = loop {
                let p = constraints.random_p_candidate();
                if let Some(progression) = constraints.progression(&p) {
                    break (p, progression);
                }
            };
            for index in [
                BigUint::from(0u8),
                &*progression.count - BigUint::from(1u8),
                OsRng.gen_biguint_below(&progression.count),
            ] {
                let q = &*progression.first + index * &progression.stride;
                assert_eq!(q.bits(), 1024);
                let n = &p * q;
                assert_eq!(n.bits(), 2048);
                assert!(
                    constraints
                        .pattern
                        .matches(&fixed_bytes::<256>(&n).unwrap()[..])
                );
            }
        }
    }
}

/// Synchronized, ordered candidate evaluation for this mode. Implementations
/// must clear secret device buffers before returning; this is not a CPU fallback.
pub type EvaluateBatch<'a> = dyn FnMut(
    &logic::rsa_modulus_vanity::RsaModulusRequest,
    &logic::hex_pattern::HexPattern,
    &[u8],
    u64,
    u32,
) -> Result<Vec<logic::candidate_result::CandidateResult>, String> + 'a;
