//! Host search orchestration independent of CUDA, also usable by differential tests.
use crate::search_control::SearchControl;
use logic::{device_search::CandidateResult, hex_pattern::HexPattern};
use zeroize::Zeroizing;

pub enum Request<'a> {
    #[cfg(feature = "p256-public-key")]
    P256Public(&'a logic::device_search::P256PublicRequest),
    #[cfg(feature = "p256-signature")]
    P256Signature(&'a logic::device_search::P256SignatureRequest),
    #[cfg(feature = "rsa-pss")]
    RsaPss(&'a logic::device_search::RsaPssRequest),
    #[cfg(feature = "rsa-modulus")]
    RsaModulus(&'a logic::device_search::RsaModulusRequest),
}

/// Implementations must synchronize before returning, preserve lane order, and
/// erase secret device allocations before releasing them. No CPU fallback.
pub trait DeviceSearch {
    fn evaluate(
        &mut self,
        request: &Request<'_>,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<Vec<CandidateResult>, String>;
}

/// Every candidate is reconstructed and verified before reserving the one output
/// slot. Fixed batches make cancellation observable between launches.
pub fn find(
    device: &mut dyn DeviceSearch,
    request: &Request<'_>,
    pattern: &HexPattern,
    message: &[u8],
    limit: u64,
    control: &SearchControl,
    mut verify: impl FnMut(u64, &[u8; 256]) -> Result<bool, String>,
) -> Result<Option<(u64, CandidateResult)>, String> {
    let stop = control.cancel_on_exit();
    while let Some(batch) = control.reserve_bounded_batch(64, limit) {
        let count = (batch.end - batch.start) as u32;
        let results =
            Zeroizing::new(device.evaluate(request, pattern, message, batch.start, count)?);
        if results.len() != count as usize {
            return Err("device returned an incorrect lane count".into());
        }
        control.add_tested(count as u64);
        for (lane, result) in results.iter().enumerate() {
            if control.stopped() {
                return Ok(None);
            }
            match result.status {
                0 => {}
                1 => {
                    let counter = batch.start + lane as u64;
                    if verify(counter, &result.bytes)? && control.claim_verified_winner() {
                        return Ok(Some((counter, *result)));
                    }
                }
                _ => return Err("device candidate evaluation failed".into()),
            }
        }
    }
    stop.finish();
    Ok(None)
}

/// Execute the exact device candidate code on the host for differential tests.
/// This type is compiled only for tests/self-test and is never a search fallback.
#[cfg(any(test, feature = "self_test"))]
pub struct HostDevice;
#[cfg(any(test, feature = "self_test"))]
impl DeviceSearch for HostDevice {
    fn evaluate(
        &mut self,
        request: &Request<'_>,
        pattern: &HexPattern,
        message: &[u8],
        start: u64,
        count: u32,
    ) -> Result<Vec<CandidateResult>, String> {
        let _ = message;
        (0..count)
            .map(|lane| {
                let counter = start
                    .checked_add(lane as u64)
                    .ok_or("test counter overflow")?;
                Ok(match request {
                    #[cfg(feature = "p256-public-key")]
                    Request::P256Public(request) => {
                        logic::device_search::p256_public(request, counter, pattern)
                    }
                    #[cfg(feature = "p256-signature")]
                    Request::P256Signature(request) => {
                        logic::device_search::p256_signature(request, message, counter, pattern)
                    }
                    #[cfg(feature = "rsa-pss")]
                    Request::RsaPss(request) => {
                        logic::device_search::rsa_pss(request, message, counter, pattern)
                    }
                    #[cfg(feature = "rsa-modulus")]
                    Request::RsaModulus(request) => {
                        logic::device_search::rsa_modulus(request, counter, pattern)
                    }
                })
            })
            .collect()
    }
}
