#![allow(unused_parens)]
#![allow(non_camel_case_types)]

use core::ops::Add;

use super::fe::Fe;
use super::gecached::GeCached;
use super::gep1p1::GeP1P1;
use super::gep2::GeP2;
use super::consts::*;

#[repr(C, align(16))]
#[derive(Clone, Copy)]
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