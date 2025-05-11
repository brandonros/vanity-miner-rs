#![allow(unused_parens)]
#![allow(non_camel_case_types)]

use core::ops::Add;

use super::fe::Fe;
use super::ge_cached::GeCached;
use super::ge_p1_p1::GeP1P1;
use super::ge_p2::GeP2;
use super::consts::*;
use super::precomputed_table::PRECOMPUTED_TABLE;

#[repr(C, align(16))]
pub struct GeP3 {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

impl GeP3 {
    pub fn new(x: Fe, y: Fe, z: Fe, t: Fe) -> GeP3 {
        GeP3 { x, y, z, t }
    }

    pub fn zero() -> GeP3 {
        GeP3 {
            x: FE_ZERO,
            y: FE_ONE,
            z: FE_ONE,
            t: FE_ZERO,
        }
    }

    fn to_cached(&self) -> GeCached {
        let y_plus_x = self.y + self.x;
        let y_minus_x = self.y - self.x;
        let z = self.z;
        let t2d = self.t * FE_D2;
        GeCached::new(y_plus_x, y_minus_x, z, t2d)
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let recip = self.z.invert();
        let x = self.x * recip;
        let y = self.y * recip;
        let x_bytes = x.to_bytes();
        let x_is_negative = x.is_negative(&x_bytes);
        let negative_flag = if x_is_negative { 1 } else { 0 };
        let y_bytes = y.to_bytes();
        let mut bs = y_bytes;
        bs[31] ^= (negative_flag << 7);
        bs
    }

    pub fn dbl(&self) -> GeP1P1 {
        self.to_p2().dbl()
    }

    fn to_p2(&self) -> GeP2 {
        GeP2::new(self.x, self.y, self.z)
    }

    pub fn ge_scalarmult_precomputed(scalar: &[u8]) -> GeP3 {
        let mut q = GeP3::zero();
        for pos in (0..=252).rev().step_by(4) {
            let byte_idx = pos >> 3;
            let bit_shift = pos & 7;
            let slot = ((scalar[byte_idx] >> bit_shift) & 15) as usize;

            let mut t = PRECOMPUTED_TABLE[0];
            for i in 1..16 {
                let other = &PRECOMPUTED_TABLE[i];
                let do_swap = (((slot ^ i).wrapping_sub(1)) >> 8) as u8 & 1;
                t.maybe_set(&other, do_swap);
            }
            q = q.add(t).to_p3();
            if pos == 0 {
                break;
            }
            q = q.dbl().to_p3();
            q = q.dbl().to_p3();
            q = q.dbl().to_p3();
            q = q.dbl().to_p3();
        }
        q
    }   
}

impl Add<GeCached> for GeP3 {
    type Output = GeP1P1;

    fn add(self, _rhs: GeCached) -> GeP1P1 {
        let y1_plus_x1 = self.y + self.x;
        let y1_minus_x1 = self.y - self.x;
        let a = y1_plus_x1 * _rhs.y_plus_x();
        let b = y1_minus_x1 * _rhs.y_minus_x();
        let c = _rhs.t2d() * self.t;
        let zz = self.z * _rhs.z();
        let d = zz + zz;
        let x3 = a - b;
        let y3 = a + b;
        let z3 = d + c;
        let t3 = d - c;
        GeP1P1::new(x3, y3, z3, t3)
    }
}

impl Add<GeP3> for GeP3 {
    type Output = GeP3;

    fn add(self, other: GeP3) -> GeP3 {
        (self + other.to_cached()).to_p3()
    }
}