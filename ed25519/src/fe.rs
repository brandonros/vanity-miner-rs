#![allow(unused_parens)]
#![allow(non_camel_case_types)]

use core::ops::{Add, Sub, Mul};

use crate::load_8u;
use super::fiat::*;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Fe(pub [u64; 5]);

impl Fe {
    pub fn new(s: [u64; 5]) -> Self {
        Self(s)
    }

    pub const fn from_bytes_const(s: &[u8; 32]) -> Fe {
        let mut h = [0u64; 5];
        let mask = 0x7ffffffffffff;
        
        h[0] = load_8u!(s, 0) & mask;
        h[1] = (load_8u!(s, 6) >> 3) & mask;
        h[2] = (load_8u!(s, 12) >> 6) & mask;
        h[3] = (load_8u!(s, 19) >> 1) & mask;
        h[4] = (load_8u!(s, 24) >> 12) & mask;
        
        Fe(h)
    }

    fn carry(&self) -> Fe {
        let mut h = Fe::new([0; 5]);
        fiat_25519_carry(&mut h.0, &self.0);
        h
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let &Fe(es) = &self.carry();
        let mut s_ = [0u8; 32];
        fiat_25519_to_bytes(&mut s_, &es);
        s_
    }

    pub fn is_negative(&self, bytes: &[u8; 32]) -> bool {
        (bytes[0] & 1) != 0
    }

    pub fn invert(&self) -> Fe {
        let z1 = *self;
        let z2 = z1.square();
        let z8 = z2.square().square();
        let z9 = z1 * z8;
        let z11 = z2 * z9;
        let z22 = z11.square();
        let z_5_0 = z9 * z22;
        let z_10_5 = (0..5).fold(z_5_0, |z_5_n, _| z_5_n.square());
        let z_10_0 = z_10_5 * z_5_0;
        let z_20_10 = (0..10).fold(z_10_0, |x, _| x.square());
        let z_20_0 = z_20_10 * z_10_0;
        let z_40_20 = (0..20).fold(z_20_0, |x, _| x.square());
        let z_40_0 = z_40_20 * z_20_0;
        let z_50_10 = (0..10).fold(z_40_0, |x, _| x.square());
        let z_50_0 = z_50_10 * z_10_0;
        let z_100_50 = (0..50).fold(z_50_0, |x, _| x.square());
        let z_100_0 = z_100_50 * z_50_0;
        let z_200_100 = (0..100).fold(z_100_0, |x, _| x.square());
        let z_200_0 = z_200_100 * z_100_0;
        let z_250_50 = (0..50).fold(z_200_0, |x, _| x.square());
        let z_250_0 = z_250_50 * z_50_0;
        let z_255_5 = (0..5).fold(z_250_0, |x, _| x.square());
        let z_255_21 = z_255_5 * z11;
        z_255_21
    }

    pub fn square(&self) -> Fe {
        let &Fe(f) = &self;
        let mut h = Fe::new([0; 5]);
        fiat_25519_carry_square(&mut h.0, f);
        h
    }

    pub fn square_and_double(&self) -> Fe {
        let h = self.square();
        let h_clone = Fe::new(h.0);
        (h + h_clone)
    }

    pub fn maybe_set(&mut self, other: &Fe, do_swap: u8) {
        let &mut Fe(f) = self;
        let &Fe(g) = other;
        let mut t = [0u64; 5];
        fiat_25519_selectznz(&mut t, do_swap, &f, &g);
        self.0 = t
    }
}

impl Add for Fe {
    type Output = Fe;

    fn add(self, _rhs: Fe) -> Fe {
        let Fe(f) = self;
        let Fe(g) = _rhs;
        let mut h = Fe::new([0; 5]);
        fiat_25519_add(&mut h.0, &f, &g);
        h
    }
}

impl Sub for Fe {
    type Output = Fe;

    fn sub(self, _rhs: Fe) -> Fe {
        let Fe(f) = self;
        let Fe(g) = _rhs;
        let mut h = Fe::new([0; 5]);
        fiat_25519_sub(&mut h.0, &f, &g);
        h.carry()
    }
}

impl Mul for Fe {
    type Output = Fe;

    fn mul(self, _rhs: Fe) -> Fe {
        let Fe(f) = self;
        let Fe(g) = _rhs;
        let mut h = Fe::new([0; 5]);
        fiat_25519_carry_mul(&mut h.0, &f, &g);
        h
    }
}