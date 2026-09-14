//! Allocation-free candidate evaluation and fixed-layout CUDA request records.
//! Host and device use the same functions. Records containing secrets must be
//! cleared by their owners after the synchronized launch; never log them.
use crate::hex_pattern::HexPattern;

/// A lane returns only public output (or an RSA factor for host validation).
/// Status: 0 = miss, 1 = match, 2 = invalid request or failed arithmetic.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CandidateResult {
    pub status: u32,
    pub bytes: [u8; 256],
}
impl CandidateResult {
    pub const MISS: Self = Self {
        status: 0,
        bytes: [0; 256],
    };
    pub const ERROR: Self = Self {
        status: 2,
        bytes: [0; 256],
    };
    pub fn matched(bytes: &[u8]) -> Self {
        let mut result = Self::MISS;
        if bytes.len() > result.bytes.len() {
            return Self::ERROR;
        }
        result.status = 1;
        result.bytes[..bytes.len()].copy_from_slice(bytes);
        result
    }
}

#[cfg(feature = "p256-public-key")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct P256PublicRequest {
    pub seed: [u8; 32],
    pub worker: u64,
    /// 0 = x, 1 = y, 2 = xy, 3 = uncompressed SEC1.
    pub target: u32,
    pub reserved: u32,
}

#[cfg(feature = "p256-public-key")]
pub fn p256_public(
    request: &P256PublicRequest,
    counter: u64,
    pattern: &HexPattern,
) -> CandidateResult {
    use crate::{
        crypto_search::{CandidateDeriver, CandidateDomain},
        p256_vanity::{PublicTarget, candidate_scalar, public_point},
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

#[cfg(feature = "p256-signature")]
#[repr(C)]
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

#[cfg(feature = "p256-signature")]
pub fn p256_signature(
    request: &P256SignatureRequest,
    message: &[u8],
    counter: u64,
    pattern: &HexPattern,
) -> CandidateResult {
    use crate::{
        crypto_search::{CandidateDeriver, CandidateDomain, hash_message_counter},
        p256_vanity::{candidate_scalar, signatures::*},
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

#[cfg(feature = "rsa-pss")]
#[repr(C)]
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

#[cfg(feature = "rsa-pss")]
pub fn rsa_pss(
    request: &RsaPssRequest,
    message: &[u8],
    counter: u64,
    pattern: &HexPattern,
) -> CandidateResult {
    use crate::{
        crypto_search::{hash_message_counter, write_salt_counter},
        rsa_crt::Rsa2048Crt,
        rsa_pss::encode_sha256,
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

#[cfg(feature = "rsa-modulus")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RsaModulusRequest {
    pub p: [u8; 128],
    /// First q in a host-constructed bounded progression, including random start.
    pub first: [u8; 128],
    pub stride: [u8; 128],
    pub upper: [u8; 128],
}

#[cfg(feature = "rsa-modulus")]
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
    if !crate::rsa_prime::probable_prime(&q) {
        return CandidateResult::MISS;
    }
    CandidateResult::matched(&q.to_be_bytes())
}

impl zeroize::Zeroize for CandidateResult {
    fn zeroize(&mut self) {
        self.status.zeroize();
        self.bytes.zeroize();
    }
}
macro_rules! clear_record {
    ($record:ty, $($field:ident),+) => {
        impl zeroize::Zeroize for $record {
            fn zeroize(&mut self) { $(self.$field.zeroize();)+ }
        }
    };
}
#[cfg(feature = "p256-public-key")]
clear_record!(P256PublicRequest, seed, worker, target, reserved);
#[cfg(feature = "p256-signature")]
clear_record!(
    P256SignatureRequest,
    private,
    seed,
    fingerprint,
    digest,
    worker,
    offset,
    length,
    source,
    target,
    s_form,
    reserved
);
#[cfg(feature = "rsa-pss")]
clear_record!(
    RsaPssRequest,
    p,
    q,
    dp,
    dq,
    q_inv,
    digest,
    salt,
    reserved,
    offset,
    length,
    source,
    salt_length
);
#[cfg(feature = "rsa-modulus")]
clear_record!(RsaModulusRequest, p, first, stride, upper);

#[cfg(test)]
mod abi_tests {
    #[test]
    fn records_have_no_implicit_padding() {
        // Transport copies complete records, so all bytes must be initialized.
        assert_eq!(core::mem::size_of::<super::CandidateResult>(), 260);
        assert_eq!(core::mem::size_of::<crate::hex_pattern::HexPattern>(), 516);
        #[cfg(feature = "p256-public-key")]
        assert_eq!(core::mem::size_of::<super::P256PublicRequest>(), 48);
        #[cfg(feature = "p256-signature")]
        assert_eq!(core::mem::size_of::<super::P256SignatureRequest>(), 168);
        #[cfg(feature = "rsa-pss")]
        assert_eq!(core::mem::size_of::<super::RsaPssRequest>(), 920);
        #[cfg(feature = "rsa-modulus")]
        assert_eq!(core::mem::size_of::<super::RsaModulusRequest>(), 512);
    }
}
