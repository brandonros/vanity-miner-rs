//! Fixed-width RSA-2048/e=65537 CRT operations for shared host/device search.
//! Hosts must establish primality, factor separation, and key validity first.
//! This module checks component consistency and every private-operation result.
//! It does not implement RSA blinding; callers must not expose it as an oracle.

use crypto_bigint::{
    Encoding, NonZero, U1024, U2048,
    modular::runtime_mod::{DynResidue, DynResidueParams},
};
use zeroize::{Zeroize, Zeroizing};

type Params1024 = DynResidueParams<{ U1024::LIMBS }>;
type Params2048 = DynResidueParams<{ U2048::LIMBS }>;

/// Secret CRT state has no Debug or serialization implementation.
pub struct Rsa2048Crt {
    p: Zeroizing<U1024>,
    q: Zeroizing<U1024>,
    dp: Zeroizing<U1024>,
    dq: Zeroizing<U1024>,
    q_inv: Zeroizing<U1024>,
    p_params: Params1024,
    q_params: Params1024,
    n: U2048,
    n_params: Params2048,
}

impl Drop for Rsa2048Crt {
    fn drop(&mut self) {
        // crypto-bigint's residue Zeroize implementation explicitly retains
        // modulus parameters. For RSA these contain secret factors. Replace our
        // cached parameters with a valid public value using volatile writes.
        // Earlier compiler temporaries/register copies cannot all be guaranteed
        // erased. The explicit scalar fields are handled by Zeroizing.
        let public = Params1024::new(&U1024::from_u8(3));
        unsafe {
            core::ptr::write_volatile(&mut self.p_params, public);
            core::ptr::write_volatile(&mut self.q_params, public);
        }
    }
}

impl Rsa2048Crt {
    /// Inputs are unsigned fixed-width big-endian components from a host-validated
    /// RSA-2048 key with two exactly 1024-bit primes. q_inv means q^-1 mod p.
    pub fn new(
        p: &[u8; 128],
        q: &[u8; 128],
        dp: &[u8; 128],
        dq: &[u8; 128],
        q_inv: &[u8; 128],
    ) -> Option<Self> {
        if p[0] & 128 == 0 || q[0] & 128 == 0 {
            return None;
        }
        let result = Self::build(p, q, dp, dq, q_inv)?;
        if result.n.to_be_bytes()[0] & 128 == 0 {
            return None;
        }
        Some(result)
    }

    fn build(
        p: &[u8; 128],
        q: &[u8; 128],
        dp: &[u8; 128],
        dq: &[u8; 128],
        q_inv: &[u8; 128],
    ) -> Option<Self> {
        if p[127] & 1 == 0 || q[127] & 1 == 0 {
            return None;
        }
        let p = Zeroizing::new(U1024::from_be_slice(p));
        let q = Zeroizing::new(U1024::from_be_slice(q));
        if *p <= U1024::ONE || *q <= U1024::ONE || p == q {
            return None;
        }
        let dp = Zeroizing::new(U1024::from_be_slice(dp));
        let dq = Zeroizing::new(U1024::from_be_slice(dq));
        let q_inv = Zeroizing::new(U1024::from_be_slice(q_inv));
        let n = p.mul(&q);
        let result = Self {
            p_params: Params1024::new(&p),
            q_params: Params1024::new(&q),
            n_params: Params2048::new(&n),
            n,
            p,
            q,
            dp,
            dq,
            q_inv,
        };
        if !result.valid_components() {
            return None;
        }
        Some(result)
    }

    fn valid_components(&self) -> bool {
        let q = DynResidue::new(&self.q, self.p_params);
        let inverse = DynResidue::new(&self.q_inv, self.p_params);
        if q.mul(&inverse).retrieve() != U1024::ONE || *self.q_inv >= *self.p {
            return false;
        }
        for (prime, exponent) in [(&self.p, &self.dp), (&self.q, &self.dq)] {
            let minus_one = Zeroizing::new(prime.wrapping_sub(&U1024::ONE));
            if **exponent == U1024::ZERO || **exponent >= *minus_one {
                return false;
            }
            let product: U2048 = exponent.mul(&U1024::from_u32(65537));
            let divisor: U2048 = minus_one.resize();
            let Some(divisor) = Option::<NonZero<U2048>>::from(NonZero::new(divisor)) else {
                return false;
            };
            if product.rem(&divisor) != U2048::ONE {
                return false;
            }
        }
        true
    }

    pub fn modulus(&self) -> [u8; 256] {
        self.n.to_be_bytes()
    }

    pub fn public_operation(&self, input: &[u8; 256]) -> Option<[u8; 256]> {
        let input = U2048::from_be_slice(input);
        if input >= self.n {
            return None;
        }
        Some(
            DynResidue::new(&input, self.n_params)
                .pow_bounded_exp(&U2048::from_u32(65537), 17)
                .retrieve()
                .to_be_bytes(),
        )
    }

    /// Return RSASP1(input), checking the public operation before releasing it.
    /// Private exponents use their full fixed width, not their secret bit length.
    pub fn private_operation(&self, input: &[u8; 256]) -> Option<[u8; 256]> {
        let representative = U2048::from_be_slice(input);
        if representative >= self.n {
            return None;
        }
        let p_wide: U2048 = self.p.resize();
        let q_wide: U2048 = self.q.resize();
        let p_nonzero = Option::<NonZero<U2048>>::from(NonZero::new(p_wide))?;
        let q_nonzero = Option::<NonZero<U2048>>::from(NonZero::new(q_wide))?;
        let p_input: U1024 = representative.rem(&p_nonzero).resize();
        let q_input: U1024 = representative.rem(&q_nonzero).resize();
        let m1 = Zeroizing::new(
            DynResidue::new(&p_input, self.p_params)
                .pow(&self.dp)
                .retrieve(),
        );
        let m2 = Zeroizing::new(
            DynResidue::new(&q_input, self.q_params)
                .pow(&self.dq)
                .retrieve(),
        );
        let difference =
            DynResidue::new(&m1, self.p_params).sub(&DynResidue::new(&m2, self.p_params));
        let mut h = difference
            .mul(&DynResidue::new(&self.q_inv, self.p_params))
            .retrieve();
        let product: U2048 = h.mul(&self.q);
        let result = product.wrapping_add(&m2.resize()).to_be_bytes();
        h.zeroize();
        if self.public_operation(&result)? != *input {
            return None;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn component(value: u64) -> [u8; 128] {
        U1024::from_u64(value).to_be_bytes()
    }
    fn textbook_key() -> Rsa2048Crt {
        // Public textbook toy RSA values, never accepted by the public constructor.
        Rsa2048Crt::build(
            &component(61),
            &component(53),
            &component(53),
            &component(49),
            &component(38),
        )
        .unwrap()
    }

    #[test]
    fn textbook_crt_and_fault_detection() {
        let mut key = textbook_key();
        for value in [0, 1, 53, 61, 65, 3232] {
            let input = U2048::from_u64(value).to_be_bytes();
            let signature = key.private_operation(&input).unwrap();
            assert_eq!(key.public_operation(&signature), Some(input));
            if value == 65 {
                assert_eq!(signature, U2048::from_u64(588).to_be_bytes());
            }
        }
        assert!(key.private_operation(&key.modulus()).is_none());
        *key.dp = U1024::ONE;
        assert!(
            key.private_operation(&U2048::from_u64(65).to_be_bytes())
                .is_none()
        );
    }

    #[test]
    fn component_validation_and_exact_width() {
        assert!(
            Rsa2048Crt::new(
                &component(61),
                &component(53),
                &component(53),
                &component(49),
                &component(38)
            )
            .is_none()
        );
        assert!(
            Rsa2048Crt::build(
                &component(60),
                &component(53),
                &component(53),
                &component(49),
                &component(38)
            )
            .is_none()
        );
        assert!(
            Rsa2048Crt::build(
                &component(61),
                &component(53),
                &component(53),
                &component(49),
                &component(37)
            )
            .is_none()
        );
        assert!(
            Rsa2048Crt::build(
                &component(61),
                &component(53),
                &component(1),
                &component(49),
                &component(38)
            )
            .is_none()
        );
    }
}
