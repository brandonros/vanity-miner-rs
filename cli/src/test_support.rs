//! Host execution adapters used only by tests and self-test.
use logic::{candidate_result::CandidateResult, hex_pattern::HexPattern};

#[cfg(feature = "p256-public-key")]
pub fn p256_public(
    request: &logic::p256_public_key_vanity::P256PublicRequest,
    pattern: &HexPattern,
    _message: &[u8],
    start: u64,
    count: u32,
) -> Result<Vec<CandidateResult>, String> {
    (0..count)
        .map(|lane| {
            let counter = start
                .checked_add(lane as u64)
                .ok_or("test counter overflow")?;
            Ok(logic::p256_public_key_vanity::p256_public(
                request, counter, pattern,
            ))
        })
        .collect()
}

#[cfg(feature = "p256-signature")]
pub fn p256_signature(
    request: &logic::p256_signature_vanity::P256SignatureRequest,
    pattern: &HexPattern,
    message: &[u8],
    start: u64,
    count: u32,
) -> Result<Vec<CandidateResult>, String> {
    (0..count)
        .map(|lane| {
            let counter = start
                .checked_add(lane as u64)
                .ok_or("test counter overflow")?;
            Ok(logic::p256_signature_vanity::p256_signature(
                request, message, counter, pattern,
            ))
        })
        .collect()
}

#[cfg(feature = "rsa-pss")]
pub fn rsa_pss(
    request: &logic::rsa_pss_signature_vanity::RsaPssRequest,
    pattern: &HexPattern,
    message: &[u8],
    start: u64,
    count: u32,
) -> Result<Vec<CandidateResult>, String> {
    (0..count)
        .map(|lane| {
            let counter = start
                .checked_add(lane as u64)
                .ok_or("test counter overflow")?;
            Ok(logic::rsa_pss_signature_vanity::rsa_pss(
                request, message, counter, pattern,
            ))
        })
        .collect()
}

#[cfg(feature = "rsa-modulus")]
pub fn rsa_modulus(
    request: &logic::rsa_modulus_vanity::RsaModulusRequest,
    pattern: &HexPattern,
    _message: &[u8],
    start: u64,
    count: u32,
) -> Result<Vec<CandidateResult>, String> {
    (0..count)
        .map(|lane| {
            let counter = start
                .checked_add(lane as u64)
                .ok_or("test counter overflow")?;
            Ok(logic::rsa_modulus_vanity::rsa_modulus(
                request, counter, pattern,
            ))
        })
        .collect()
}
