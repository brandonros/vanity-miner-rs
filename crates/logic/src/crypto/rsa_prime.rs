//! Device-side RSA candidate sieve and Miller-Rabin filter. A host must repeat
//! primality testing with fresh random bases before accepting any key.
use crypto_bigint::{
    U1024,
    modular::runtime_mod::{DynResidue, DynResidueParams},
};

/// Miller-Rabin bases, the same on every target so CPU and GPU agree. Twelve
/// fixed bases is a filter, not a proof: each extra round is a full 1024-bit
/// exponentiation on a batch's slowest lane, and a hit costs one host-side
/// verification with fresh random bases, which is always performed.
const BASES: [u32; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// Trial-division depth: the first 512 primes (through 3671).
const SIEVE_PRIMES: usize = 512;
const SIEVE_GROUP_MAX: usize = 8;
/// Group products stay below 2^32, so a remainder folded in 32-bit chunks
/// never leaves u64. There is deliberately no u128 path: Metal has no 128-bit
/// integers, and a u128 remainder would have to be legalized as a bit loop.
const SIEVE_PRODUCT_LIMIT: u64 = 1 << 32;

struct SieveGroup {
    product: u32,
    primes: [u32; SIEVE_GROUP_MAX],
    len: u8,
}

const fn sieve_primes() -> [u32; SIEVE_PRIMES] {
    let mut primes = [0u32; SIEVE_PRIMES];
    let mut count = 0usize;
    let mut candidate = 2u32;
    while count < SIEVE_PRIMES {
        let mut prime = true;
        let mut i = 0usize;
        while i < count {
            let p = primes[i];
            if (p as u64) * (p as u64) > candidate as u64 {
                break;
            }
            if candidate % p == 0 {
                prime = false;
                break;
            }
            i += 1;
        }
        if prime {
            primes[count] = candidate;
            count += 1;
        }
        candidate += 1;
    }
    primes
}

const fn sieve_groups() -> ([SieveGroup; SIEVE_PRIMES], usize) {
    const EMPTY: SieveGroup = SieveGroup {
        product: 0,
        primes: [0; SIEVE_GROUP_MAX],
        len: 0,
    };
    let primes = sieve_primes();
    let mut groups = [EMPTY; SIEVE_PRIMES];
    let mut group = 0usize;
    let mut product: u64 = 1;
    let mut len = 0usize;
    let mut i = 0usize;
    while i < SIEVE_PRIMES {
        if len == SIEVE_GROUP_MAX || product * (primes[i] as u64) >= SIEVE_PRODUCT_LIMIT {
            groups[group].product = product as u32;
            groups[group].len = len as u8;
            group += 1;
            product = 1;
            len = 0;
        }
        groups[group].primes[len] = primes[i];
        product *= primes[i] as u64;
        len += 1;
        i += 1;
    }
    groups[group].product = product as u32;
    groups[group].len = len as u8;
    (groups, group + 1)
}

const SIEVE: ([SieveGroup; SIEVE_PRIMES], usize) = sieve_groups();

/// n mod a nonzero divisor below 2^32, folded in 32-bit chunks: the running
/// remainder stays below 2^32, so `remainder * 2^32 + chunk` fits u64. One word
/// division per chunk instead of a full-width rem(). A zero or oversized divisor
/// returns 0, which callers treat as "divisible" and therefore reject.
fn rem_small(n: &U1024, divisor: u32) -> u64 {
    let Some(divisor) = core::num::NonZeroU64::new(u64::from(divisor)) else {
        return 0;
    };
    let mut remainder = 0u64;
    for limb in n.as_limbs().iter().rev() {
        remainder = ((remainder << 32) | (limb.0 >> 32)) % divisor;
        remainder = ((remainder << 32) | (limb.0 & 0xffff_ffff)) % divisor;
    }
    remainder
}

/// Fixed-base probable-prime filtering, not a proof or adversarial-key validator.
/// Includes gcd(q-1, 65537) and small-prime screening before exponentiation.
pub fn probable_prime(n: &U1024) -> bool {
    if *n < U1024::from_u8(2) {
        return false;
    }
    // State the Montgomery modulus precondition directly. The sieve below also
    // rejects even composites, but that implication crosses a division loop.
    if n.as_limbs()[0].0 & 1 == 0 {
        return *n == U1024::from_u8(2);
    }
    for group in SIEVE.0.iter().take(SIEVE.1) {
        let remainder = rem_small(n, group.product);
        for &prime in group.primes.iter().take(group.len as usize) {
            let Some(divisor) = core::num::NonZeroU64::new(u64::from(prime)) else {
                return false;
            };
            if remainder % divisor == 0 {
                return *n == U1024::from_u32(prime);
            }
        }
    }
    let minus_one = n.wrapping_sub(&U1024::ONE);
    if rem_small(&minus_one, 65537) == 0 {
        return false;
    }
    let s = minus_one.trailing_zeros();
    let d = minus_one.shr_vartime(s);
    let params = DynResidueParams::new(n);
    for base in BASES {
        let mut x = DynResidue::new(&U1024::from_u32(base), params).pow(&d);
        let value = x.retrieve();
        if value == U1024::ONE || value == minus_one {
            continue;
        }
        let mut passed = false;
        for _ in 1..s {
            x = x.square();
            let value = x.retrieve();
            if value == minus_one {
                passed = true;
                break;
            }
            if value == U1024::ONE {
                return false;
            }
        }
        if !passed {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prime_filter_rejects_carmichael_and_strong_pseudoprimes() {
        for n in [2, 3, 131, 137, 65537, 104729] {
            assert!(probable_prime(&U1024::from_u64(n)));
        }
        for n in [
            0,
            1,
            4,
            561,
            1105,
            1729,
            3215031751,
            341550071728321,
            3825123056546413051,
        ] {
            assert!(!probable_prime(&U1024::from_u64(n)));
        }
    }
}
