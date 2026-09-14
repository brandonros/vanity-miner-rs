//! Host execution adapters used only by tests and self-test.
use logic::{
    search::candidate_result::{BatchResult, CandidateResult},
    search::hex_pattern::HexPattern,
};

fn evaluate(
    start: u64,
    count: u32,
    mut candidate: impl FnMut(u64) -> CandidateResult,
) -> Result<BatchResult, String> {
    let mut output = BatchResult::EMPTY;
    for lane in 0..count {
        let counter = start
            .checked_add(lane as u64)
            .ok_or("test counter overflow")?;
        let result = candidate(counter);
        match result.status {
            0 => {}
            1 => {
                if output.matches == 0 {
                    output.lane = lane;
                    output.candidate = result;
                }
                output.matches += 1;
            }
            _ => output.errors += 1,
        }
    }
    Ok(output)
}

#[cfg(feature = "p256-public-key")]
pub fn p256_public(
    request: &logic::modes::p256_public_key_vanity::P256PublicRequest,
    pattern: &HexPattern,
    _message: &[u8],
    start: u64,
    count: u32,
) -> Result<BatchResult, String> {
    evaluate(start, count, |counter| {
        logic::modes::p256_public_key_vanity::p256_public(request, counter, pattern)
    })
}

#[cfg(feature = "p256-signature")]
pub fn p256_signature(
    request: &logic::modes::p256_signature_vanity::P256SignatureRequest,
    pattern: &HexPattern,
    message: &[u8],
    start: u64,
    count: u32,
) -> Result<BatchResult, String> {
    evaluate(start, count, |counter| {
        logic::modes::p256_signature_vanity::p256_signature(request, message, counter, pattern)
    })
}

#[cfg(feature = "rsa-pss")]
pub fn rsa_pss(
    request: &logic::modes::rsa_pss_signature_vanity::RsaPssRequest,
    pattern: &HexPattern,
    message: &[u8],
    start: u64,
    count: u32,
) -> Result<BatchResult, String> {
    evaluate(start, count, |counter| {
        logic::modes::rsa_pss_signature_vanity::rsa_pss(request, message, counter, pattern)
    })
}

#[cfg(feature = "rsa-modulus")]
pub fn rsa_modulus(
    request: &logic::modes::rsa_modulus_vanity::RsaModulusRequest,
    pattern: &HexPattern,
    _message: &[u8],
    start: u64,
    count: u32,
) -> Result<BatchResult, String> {
    evaluate(start, count, |counter| {
        logic::modes::rsa_modulus_vanity::rsa_modulus(request, counter, pattern)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_counts_all_matches_but_keeps_only_first_payload() {
        let result = evaluate(10, 4, |counter| {
            if counter == 10 {
                CandidateResult::MISS
            } else {
                CandidateResult::matched(&[counter as u8])
            }
        })
        .unwrap();
        assert_eq!(result.matches, 3);
        let (lane, candidate) = result.winner(4).unwrap().unwrap();
        assert_eq!(lane, 1);
        assert_eq!(candidate.bytes[0], 11);
    }

    #[test]
    fn batch_still_observes_errors_after_winning_lane() {
        let result = evaluate(0, 2, |counter| {
            if counter == 0 {
                CandidateResult::matched(&[42])
            } else {
                CandidateResult::ERROR
            }
        })
        .unwrap();
        assert_eq!(result.matches, 1);
        assert_eq!(result.errors, 1);
        assert!(result.winner(2).is_err());
    }
}

#[cfg(test)]
pub fn console_field(record: &str, name: &str) -> Vec<u8> {
    let value = record
        .lines()
        .find_map(|line| {
            let (_, field) = line.split_once("] ")?;
            let (key, value) = field.split_once('=')?;
            (key == name).then_some(value)
        })
        .expect("missing console output field");
    hex::decode(value).unwrap()
}
