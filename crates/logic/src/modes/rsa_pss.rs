//! Candidate evaluation and device request layout for rsa-pss.
//! Owners must clear secret records after synchronized device use.
use crate::{search::candidate_result::CandidateResult, search::hex_pattern::HexPattern};

llvm_metal_kernel::record! {
#[derive(Clone, Copy)]
pub struct RsaPssRequest {
    pub p: [u8; 128],
    pub q: [u8; 128],
    pub dp: [u8; 128],
    pub dq: [u8; 128],
    pub q_inv: [u8; 128],
    pub digest: [u8; 32],
    pub salt: [u8; 222],
    pub reserved: [u8; 2],
    pub offset: u64,
    pub length: u64,
    /// 0 = enumerate salts, 1 = enumerate message window with fixed salt.
    pub source: u32,
    pub salt_length: u32,
}
}

pub fn rsa_pss(
    request: &RsaPssRequest,
    message: &[u8],
    counter: u64,
    pattern: &HexPattern,
) -> CandidateResult {
    use crate::{
        crypto::rsa_crt::Rsa2048Crt,
        crypto::rsa_pss::encode_sha256,
        search::{message_window::hash_message_counter, salt_counter::write_salt_counter},
    };
    let length = request.salt_length as usize;
    if length > 222 {
        return CandidateResult::ERROR;
    }
    let mut salt = [0; 222];
    let digest = match request.source {
        0 => {
            if write_salt_counter(&request.salt[..length], counter, &mut salt[..length]).is_err() {
                return CandidateResult::ERROR;
            }
            request.digest
        }
        1 => {
            salt[..length].copy_from_slice(&request.salt[..length]);
            let (Ok(offset), Ok(length)) = (
                usize::try_from(request.offset),
                usize::try_from(request.length),
            ) else {
                return CandidateResult::ERROR;
            };
            let Ok(digest) = hash_message_counter(message, offset, length, counter as u128) else {
                return CandidateResult::ERROR;
            };
            digest
        }
        _ => return CandidateResult::ERROR,
    };
    let mut encoded = [0; 256];
    if encode_sha256(&digest, &salt[..length], 2047, &mut encoded).is_err() {
        return CandidateResult::ERROR;
    }
    let Some(key) = Rsa2048Crt::new(
        &request.p,
        &request.q,
        &request.dp,
        &request.dq,
        &request.q_inv,
    ) else {
        return CandidateResult::ERROR;
    };
    let Some(signature) = key.private_operation(&encoded) else {
        return CandidateResult::ERROR;
    };
    if pattern.matches(&signature) {
        CandidateResult::matched(&signature)
    } else {
        CandidateResult::MISS
    }
}

impl zeroize::Zeroize for RsaPssRequest {
    fn zeroize(&mut self) {
        self.p.zeroize();
        self.q.zeroize();
        self.dp.zeroize();
        self.dq.zeroize();
        self.q_inv.zeroize();
        self.digest.zeroize();
        self.salt.zeroize();
        self.reserved.zeroize();
        self.offset.zeroize();
        self.length.zeroize();
        self.source.zeroize();
        self.salt_length.zeroize();
    }
}

// SAFETY: repr(C), padding-free integer fields and arrays; all bit patterns are valid.
unsafe impl crate::search::device_record::DeviceRecord for RsaPssRequest {}
