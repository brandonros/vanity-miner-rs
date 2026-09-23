//! On-device / on-CPU self-test: runs known-answer tests for every logic
//! primitive against externally-validated expected values, writing
//! pass(1)/fail(0) per check into the results buffer.
//!
//! Each slot has a dedicated `register_self_test!` function. GPU mode runs one kernel per
//! mode, using generated result indices. CPU mode calls all checks in sequence.
//!
//! Keep known-answer inputs opaque before the operation under test. A barrier
//! around the final boolean is too late: the operation can already be folded.
//! `black_box` is best effort; inspect emitted PTX to verify the computation
//! survives optimization.

#[macro_use]
mod registration;
include!("registry.rs");

#[cfg(any(
    feature = "self_test_p256_public_key",
    feature = "self_test_p256_signature",
    feature = "self_test_rsa_pss"
))]
mod known_answers;

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
pub(super) fn bytes_eq_prefix(actual: &[u8; 64], expected: &[u8]) -> bool {
    let n = expected.len();
    let mut i = 0;
    while i < n {
        if actual[i] != expected[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(any(
    feature = "self_test_p256_public_key",
    feature = "self_test_p256_signature",
    feature = "self_test_rsa_pss"
))]
fn record_candidate(
    h: &mut crate::crypto::sha256::Sha256,
    result: crate::search::candidate_result::CandidateResult,
) {
    h.update(result.status.to_le_bytes());
    h.update(result.bytes);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn enabled_self_test_checks_pass_on_cpu_and_disabled_slots_are_untouched() {
        const UNWRITTEN: u32 = 0xa5a5a5a5;
        let mut results = [UNWRITTEN; SELF_TEST_NUM_CHECKS];
        run_self_test(&mut results);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(
                r,
                if metadata::CASES[i].enabled {
                    1
                } else {
                    UNWRITTEN
                },
                "self-test check {} ({}) failed",
                i,
                metadata::CASES[i].name
            );
        }
    }
}
