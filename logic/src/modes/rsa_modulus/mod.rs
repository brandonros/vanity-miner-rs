//! Candidate evaluation and CUDA request layout for rsa-modulus.
//! Owners must clear secret records after synchronized device use.
use crate::{search::candidate_result::CandidateResult, search::hex_pattern::HexPattern};

pub mod pipeline;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsaModulusRequest {
    /// 0: test q against p/pattern; 1: filter independent p candidates.
    pub stage: u32,
    pub reserved: u32,
    pub p: [u8; 128],
    /// First q in a host-constructed bounded progression, including random start.
    pub first: [u8; 128],
    pub stride: [u8; 128],
    pub upper: [u8; 128],
}

pub fn rsa_modulus(
    request: &RsaModulusRequest,
    counter: u64,
    pattern: &HexPattern,
) -> CandidateResult {
    use crypto_bigint::{Encoding, U1024, U2048};
    use zeroize::Zeroizing;
    let first = Zeroizing::new(U1024::from_be_slice(&request.first));
    let stride = Zeroizing::new(U1024::from_be_slice(&request.stride));
    if *stride == U1024::ZERO {
        return CandidateResult::ERROR;
    }
    let step: U2048 = stride.mul(&U1024::from_u64(counter));
    let q_wide = Zeroizing::new(step.wrapping_add(&first.resize()));
    let upper: U2048 = U1024::from_be_slice(&request.upper).resize();
    if *q_wide > upper {
        return CandidateResult::ERROR;
    }
    let q: Zeroizing<U1024> = Zeroizing::new(q_wide.resize());
    if request.stage == 1 {
        if q.bits() != 1024 || !crate::crypto::rsa_prime::probable_prime(&q) {
            return CandidateResult::MISS;
        }
        return CandidateResult::matched(&q.to_be_bytes());
    }
    if request.stage != 0 {
        return CandidateResult::ERROR;
    }
    let p = Zeroizing::new(U1024::from_be_slice(&request.p));
    if p.bits() != 1024 || q.bits() != 1024 || p == q {
        return CandidateResult::MISS;
    }
    let distance = if *p > *q {
        p.wrapping_sub(&q)
    } else {
        q.wrapping_sub(&p)
    };
    if distance <= U1024::ONE.shl_vartime(924) {
        return CandidateResult::MISS;
    }
    let modulus: U2048 = p.mul(&q);
    if modulus.bits() != 2048 || !pattern.matches(&modulus.to_be_bytes()) {
        return CandidateResult::MISS;
    }
    if !crate::crypto::rsa_prime::probable_prime(&q) {
        return CandidateResult::MISS;
    }
    CandidateResult::matched(&q.to_be_bytes())
}

impl zeroize::Zeroize for RsaModulusRequest {
    fn zeroize(&mut self) {
        self.stage.zeroize();
        self.reserved.zeroize();
        self.p.zeroize();
        self.first.zeroize();
        self.stride.zeroize();
        self.upper.zeroize();
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn request_has_no_implicit_padding() {
        // CUDA transport copies the complete initialized request record.
        assert_eq!(core::mem::size_of::<super::RsaModulusRequest>(), 520);
    }
}

// SAFETY: repr(C), padding-free integer fields and arrays; all bit patterns are valid.
unsafe impl crate::search::device_record::DeviceRecord for RsaModulusRequest {}
