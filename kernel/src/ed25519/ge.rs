#![allow(unused_parens)]
#![allow(non_camel_case_types)]

use core::ops::Add as _;

use super::gep3::GeP3;
use super::precomputed_table::PRECOMPUTED_TABLE;
pub struct Ge;

impl Ge {
    pub fn ge_scalarmult_precomputed(scalar: &[u8]) -> GeP3 {
        let mut q = GeP3::zero();
        for pos in (0..=252).rev().step_by(4) {
            let slot = ((scalar[pos >> 3] >> (pos & 7)) & 15) as usize;
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
