//! Device-side RSA candidate sieve and Miller-Rabin filter. A host must repeat
//! primality testing with fresh random bases before accepting any key.
use crypto_bigint::{
    NonZero, U1024,
    modular::runtime_mod::{DynResidue, DynResidueParams},
};

const BASES: [u32; 32] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131,
];

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
    for prime in BASES {
        let small = U1024::from_u32(prime);
        if *n == small {
            return true;
        }
        let Some(prime) = core::num::NonZeroU32::new(prime) else {
            return false;
        };
        let divisor = NonZero::<U1024>::from_u32(prime);
        if n.rem(&divisor) == U1024::ZERO {
            return false;
        }
    }
    let minus_one = n.wrapping_sub(&U1024::ONE);
    let e = NonZero::from_uint(U1024::from_u32(65537));
    if minus_one.rem(&e) == U1024::ZERO {
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
