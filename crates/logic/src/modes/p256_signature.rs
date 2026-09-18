//! Candidate evaluation and device request layout for p256-signature.
//! Owners must clear secret records after synchronized device use.
use crate::{search::candidate_result::CandidateResult, search::hex_pattern::HexPattern};

llvm_metal_kernel::record! {
#[derive(Clone, Copy)]
pub struct P256SignatureRequest {
    pub private: [u8; 32],
    pub seed: [u8; 32],
    pub fingerprint: [u8; 32],
    pub digest: [u8; 32],
    pub worker: u64,
    pub offset: u64,
    pub length: u64,
    /// 0 = mutable message with RFC6979, 1 = secret ephemeral search.
    pub source: u32,
    /// 0 = raw, 1 = r, 2 = s.
    pub target: u32,
    /// 0 = low, 1 = high, 2 = either.
    pub s_form: u32,
    pub reserved: u32,
}
}

pub fn p256_signature(
    request: &P256SignatureRequest,
    message: &[u8],
    counter: u64,
    pattern: &HexPattern,
) -> CandidateResult {
    use crate::{
        crypto::p256::{candidate_scalar, signatures::*},
        search::{
            candidate_derivation::{CandidateDeriver, CandidateDomain},
            message_window::hash_message_counter,
        },
    };
    let target = match request.target {
        0 => SignatureTarget::Raw,
        1 => SignatureTarget::R,
        2 => SignatureTarget::S,
        _ => return CandidateResult::ERROR,
    };
    let form = match request.s_form {
        0 => SForm::Low,
        1 => SForm::High,
        2 => SForm::Either,
        _ => return CandidateResult::ERROR,
    };
    let matched = match request.source {
        0 => {
            let (Ok(offset), Ok(length)) = (
                usize::try_from(request.offset),
                usize::try_from(request.length),
            ) else {
                return CandidateResult::ERROR;
            };
            let Ok(digest) = hash_message_counter(message, offset, length, counter as u128) else {
                return CandidateResult::ERROR;
            };
            let Some(raw) = sign_digest(&request.private, &digest) else {
                return CandidateResult::ERROR;
            };
            matching_representation(&raw, target, form, pattern)
        }
        1 => {
            let deriver = CandidateDeriver::new(
                request.seed,
                CandidateDomain::P256Ephemeral,
                request.fingerprint,
                request.digest,
            );
            let Some(nonce) = candidate_scalar(&deriver, request.worker, counter as u128) else {
                return CandidateResult::ERROR;
            };
            matching_ephemeral_signature(
                &request.private,
                &request.digest,
                &nonce,
                target,
                form,
                pattern,
            )
        }
        _ => return CandidateResult::ERROR,
    };
    matched.map_or(CandidateResult::MISS, |raw| CandidateResult::matched(&raw))
}

impl zeroize::Zeroize for P256SignatureRequest {
    fn zeroize(&mut self) {
        self.private.zeroize();
        self.seed.zeroize();
        self.fingerprint.zeroize();
        self.digest.zeroize();
        self.worker.zeroize();
        self.offset.zeroize();
        self.length.zeroize();
        self.source.zeroize();
        self.target.zeroize();
        self.s_form.zeroize();
        self.reserved.zeroize();
    }
}

// SAFETY: repr(C), padding-free integer fields and arrays; all bit patterns are valid.
unsafe impl crate::search::device_record::DeviceRecord for P256SignatureRequest {}
