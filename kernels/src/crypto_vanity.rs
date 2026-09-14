//! Thin CUDA adapters: each lane evaluates one distinct candidate, returning a
//! fixed-size result. Host synchronization precedes reading or erasing buffers.
use cuda_std::prelude::*;
use logic::{device_search::CandidateResult, hex_pattern::HexPattern};

/// The host supplies valid, aligned request/pattern pointers, count writable
/// result records, and message_len readable bytes. Launches may overprovision
/// lanes. Requests and inputs remain immutable until stream synchronization.
macro_rules! candidate_kernel {
    ($name:ident, $request:ty, $evaluate:path) => {
        #[kernel]
        #[allow(clippy::missing_safety_doc)]
        pub unsafe extern "C" fn $name(
            request: *const $request,
            pattern: *const HexPattern,
            message: *const u8,
            message_len: usize,
            start: u64,
            count: u32,
            results: *mut CandidateResult,
        ) {
            let lane = crate::utilities::get_thread_idx();
            if lane >= count as usize {
                return;
            }
            let result = if let Some(counter) = start.checked_add(lane as u64) {
                let message = if message_len == 0 {
                    &[]
                } else {
                    unsafe { core::slice::from_raw_parts(message, message_len) }
                };
                unsafe { $evaluate(&*request, message, counter, &*pattern) }
            } else {
                CandidateResult::ERROR
            };
            unsafe {
                results.add(lane as usize).write(result);
            }
        }
    };
}

#[cfg(feature = "p256-public-key")]
fn public(
    request: &logic::device_search::P256PublicRequest,
    _: &[u8],
    counter: u64,
    pattern: &HexPattern,
) -> CandidateResult {
    logic::device_search::p256_public(request, counter, pattern)
}
#[cfg(feature = "rsa-modulus")]
fn modulus(
    request: &logic::device_search::RsaModulusRequest,
    _: &[u8],
    counter: u64,
    pattern: &HexPattern,
) -> CandidateResult {
    logic::device_search::rsa_modulus(request, counter, pattern)
}

#[cfg(feature = "p256-public-key")]
candidate_kernel!(
    kernel_p256_public_key_vanity,
    logic::device_search::P256PublicRequest,
    public
);
#[cfg(feature = "p256-signature")]
candidate_kernel!(
    kernel_p256_signature_vanity,
    logic::device_search::P256SignatureRequest,
    logic::device_search::p256_signature
);
#[cfg(feature = "rsa-pss")]
candidate_kernel!(
    kernel_rsa_pss_signature_vanity,
    logic::device_search::RsaPssRequest,
    logic::device_search::rsa_pss
);
#[cfg(feature = "rsa-modulus")]
candidate_kernel!(
    kernel_rsa_modulus_vanity,
    logic::device_search::RsaModulusRequest,
    modulus
);
