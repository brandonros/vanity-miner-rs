//! Candidate evaluation and device request layout for p256-public-key.
//! Owners must clear secret records after synchronized device use.
use crate::{search::candidate_result::CandidateResult, search::hex_pattern::HexPattern};

llvm_metal_kernel::record! {
#[derive(Clone, Copy)]
pub struct P256PublicRequest {
    pub seed: [u8; 32],
    pub worker: u64,
    /// 0 = x, 1 = y, 2 = xy, 3 = uncompressed SEC1.
    pub target: u32,
    pub reserved: u32,
}
}

pub fn p256_public(
    request: &P256PublicRequest,
    counter: u64,
    pattern: &HexPattern,
) -> CandidateResult {
    use crate::{
        crypto::p256::{PublicTarget, candidate_scalar, public_point},
        search::candidate_derivation::{CandidateDeriver, CandidateDomain},
    };
    let target = match request.target {
        0 => PublicTarget::X,
        1 => PublicTarget::Y,
        2 => PublicTarget::Xy,
        3 => PublicTarget::Uncompressed,
        _ => return CandidateResult::ERROR,
    };
    let deriver = CandidateDeriver::new(
        request.seed,
        CandidateDomain::P256PrivateKey,
        [0; 32],
        [0; 32],
    );
    let Some(private) = candidate_scalar(&deriver, request.worker, counter as u128) else {
        return CandidateResult::ERROR;
    };
    let Some(point) = public_point(&private) else {
        return CandidateResult::ERROR;
    };
    if pattern.matches(target.bytes(&point)) {
        CandidateResult::matched(&point)
    } else {
        CandidateResult::MISS
    }
}

impl zeroize::Zeroize for P256PublicRequest {
    fn zeroize(&mut self) {
        self.seed.zeroize();
        self.worker.zeroize();
        self.target.zeroize();
        self.reserved.zeroize();
    }
}

// SAFETY: repr(C), padding-free integer fields and arrays; all bit patterns are valid.
unsafe impl crate::search::device_record::DeviceRecord for P256PublicRequest {}
