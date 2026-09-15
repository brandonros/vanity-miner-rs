use super::*;

pub struct ModulusConstraints {
    pub pattern: HexPattern,
    pub(super) lower: BigUint,
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
    pub(super) fn search_budget(&self) -> u64 {
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

pub(super) fn ceil_div(n: &BigUint, d: &BigUint) -> BigUint {
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
        // HexPattern checks width and contradictory overlaps. Narrow or empty
        // per-factor ranges are ordinary search outcomes, not a prefix-length
        // limit. A full-width suffix fixes at most one q for each odd p.
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
    pub(super) fn random_p_candidate(&self) -> BigUint {
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

    /// Search-wide constants only. No per-factor preparation is done by the host.
    pub fn device_config(
        &self,
        seed: [u8; 32],
        worker: u64,
    ) -> Result<logic::modes::rsa_modulus::pipeline::SearchConfig, String> {
        let one = BigUint::from(1u8);
        let max = (&one << 1024usize) - &one;
        let mut min = ceil_div(&self.lower, &max).max(&one << 1023usize);
        if &min % 2u8 == BigUint::from(0u8) {
            min += &one;
        }
        let count = (&max - &min) / 2u8 + &one;
        Ok(logic::modes::rsa_modulus::pipeline::SearchConfig {
            lower: *fixed_bytes(&self.lower)?,
            upper: *fixed_bytes(&self.upper)?,
            p_min: *fixed_bytes(&min)?,
            p_count: *fixed_bytes(&count)?,
            suffix: *fixed_bytes(&self.suffix)?,
            seed,
            worker,
            suffix_bits: self.suffix_bits as u32,
            reserved: 0,
        })
    }
}
