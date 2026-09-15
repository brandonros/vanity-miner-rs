//! On-device / on-CPU self-test: runs known-answer tests for every logic
//! primitive against externally-validated expected values, writing
//! pass(1)/fail(0) per check into the results buffer.
//!
//! Each slot has a dedicated `check_*` function. GPU mode runs eight mode-specific
//! kernels, preserving individual result slots. CPU mode calls all checks in sequence.
//!
//! Keep known-answer inputs opaque before the operation under test. A barrier
//! around the final boolean is too late: the operation can already be folded.
//! `black_box` is best effort; inspect emitted PTX to verify the computation
//! survives optimization.

#[cfg(feature = "self_test_bitcoin")]
pub mod bitcoin;
#[cfg(feature = "self_test_ethereum")]
pub mod ethereum;
#[cfg(any(
    feature = "self_test_p256_public_key",
    feature = "self_test_p256_signature",
    feature = "self_test_rsa_pss",
    feature = "self_test_rsa_modulus"
))]
mod known_answers;
#[cfg(feature = "self_test_p256_public_key")]
pub mod p256_public_key;
#[cfg(feature = "self_test_p256_signature")]
pub mod p256_signature;
#[cfg(feature = "self_test_rsa_modulus")]
pub mod rsa_modulus;
#[cfg(feature = "self_test_rsa_pss")]
pub mod rsa_pss;
#[cfg(feature = "self_test_shallenge")]
pub mod shallenge;
#[cfg(feature = "self_test_solana")]
pub mod solana;

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
    feature = "self_test_rsa_pss",
    feature = "self_test_rsa_modulus"
))]
fn record_candidate(
    h: &mut crate::crypto::sha256::Sha256,
    result: crate::search::candidate_result::CandidateResult,
) {
    h.update(result.status.to_le_bytes());
    h.update(result.bytes);
}

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
pub struct IdxProbe(pub [u64; 5]);

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
impl core::ops::Index<usize> for IdxProbe {
    type Output = u64;
    fn index(&self, i: usize) -> &u64 {
        &(self.0[i])
    }
}

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
impl core::ops::IndexMut<usize> for IdxProbe {
    fn index_mut(&mut self, i: usize) -> &mut u64 {
        &mut (self.0[i])
    }
}

pub const SELF_TEST_NUM_CHECKS: usize = 160;

pub mod metadata;
pub const SELF_TEST_LABELS: [&str; SELF_TEST_NUM_CHECKS] = metadata::labels();

pub fn run_self_test(results: &mut [u32]) {
    #[cfg(feature = "self_test_solana")]
    solana::run(results);
    #[cfg(feature = "self_test_bitcoin")]
    bitcoin::run(results);
    #[cfg(feature = "self_test_ethereum")]
    ethereum::run(results);
    #[cfg(feature = "self_test_shallenge")]
    shallenge::run(results);
    #[cfg(feature = "self_test_p256_public_key")]
    p256_public_key::run(results);
    #[cfg(feature = "self_test_p256_signature")]
    p256_signature::run(results);
    #[cfg(feature = "self_test_rsa_pss")]
    rsa_pss::run(results);
    #[cfg(feature = "self_test_rsa_modulus")]
    rsa_modulus::run(results);
}

#[cfg(all(test, feature = "self_test"))]
mod test {
    use super::*;

    #[test]
    fn all_self_test_checks_pass_on_cpu() {
        let mut results = [0u32; SELF_TEST_NUM_CHECKS];
        run_self_test(&mut results);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(
                r, 1,
                "self-test check {} ({}) failed",
                i, SELF_TEST_LABELS[i]
            );
        }
    }
}
