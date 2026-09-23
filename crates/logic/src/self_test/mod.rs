//! Known-answer self-tests for every logic primitive, on CPU and on device.
//!
//! Each mode is one file with one `checks!` block. A check returns `true` on
//! the expected answer; a mode's `run` writes one `u32` per check in
//! declaration order, so a self-test kernel's results buffer is
//! `[u32; CHECKS.len()]` and no index is shared between modes.
//!
//! Keep known-answer inputs opaque before the operation under test. A barrier
//! around the final boolean is too late: the operation can already be folded.
//! `black_box` is best effort; inspect emitted PTX to verify the computation
//! survives optimization.

pub struct Check {
    pub name: &'static str,
    /// The doc comment of the check.
    pub label: &'static str,
    pub run: fn() -> bool,
}

pub struct Mode {
    pub name: &'static str,
    pub checks: &'static [Check],
    /// Writes `checks.len()` results: 1 for a pass, 0 for a failure.
    pub run: fn(&mut [u32]),
}

/// Declares a mode's checks: the functions, `CHECKS`, and `run`.
macro_rules! checks {
    ($($(#[doc = $label:literal])+ fn $name:ident() -> bool $body:block)*) => {
        $(
            $(#[doc = $label])+
            #[inline(never)]
            pub fn $name() -> bool $body
        )*

        pub const CHECKS: &[super::Check] = &[$(super::Check {
            name: stringify!($name),
            label: concat!($($label, "\n"),+).trim_ascii(),
            run: $name,
        },)*];

        /// Direct calls: the device build carries no function pointers.
        pub fn run(results: &mut [u32]) {
            let mut results = results.iter_mut();
            $(*results.next().expect("results buffer too short") = $name() as u32;)*
        }
    };
}

#[cfg(feature = "self_test_bitcoin")]
pub mod bitcoin;
#[cfg(feature = "self_test_ethereum")]
pub mod ethereum;
#[cfg(feature = "self_test_p256_public_key")]
pub mod p256_public_key;
#[cfg(feature = "self_test_p256_signature")]
pub mod p256_signature;
#[cfg(feature = "self_test_rsa_pss")]
pub mod rsa_pss;
#[cfg(feature = "self_test_shallenge")]
pub mod shallenge;
#[cfg(feature = "self_test_solana")]
pub mod solana;

/// Every compiled-in mode. A mode's kernel is `kernel_self_test_<name>` in
/// `self_test_<name>.ptx`.
pub const MODES: &[Mode] = &[
    #[cfg(feature = "self_test_solana")]
    Mode {
        name: "solana",
        checks: solana::CHECKS,
        run: solana::run,
    },
    #[cfg(feature = "self_test_bitcoin")]
    Mode {
        name: "bitcoin",
        checks: bitcoin::CHECKS,
        run: bitcoin::run,
    },
    #[cfg(feature = "self_test_ethereum")]
    Mode {
        name: "ethereum",
        checks: ethereum::CHECKS,
        run: ethereum::run,
    },
    #[cfg(feature = "self_test_shallenge")]
    Mode {
        name: "shallenge",
        checks: shallenge::CHECKS,
        run: shallenge::run,
    },
    #[cfg(feature = "self_test_p256_public_key")]
    Mode {
        name: "p256_public_key",
        checks: p256_public_key::CHECKS,
        run: p256_public_key::run,
    },
    #[cfg(feature = "self_test_p256_signature")]
    Mode {
        name: "p256_signature",
        checks: p256_signature::CHECKS,
        run: p256_signature::run,
    },
    #[cfg(feature = "self_test_rsa_pss")]
    Mode {
        name: "rsa_pss",
        checks: rsa_pss::CHECKS,
        run: rsa_pss::run,
    },
];

/// The `candidate` entry of a vanity mode, driven from the batch that reaches
/// a known answer: rng seed `seed` and lane `thread_idx` of the known key are
/// batch seed `seed - 1` at counter `32 + thread_idx`.
#[cfg(any(
    feature = "self_test_solana",
    feature = "self_test_bitcoin",
    feature = "self_test_ethereum"
))]
mod candidate {
    use crate::search::{
        candidate_result::CandidateResult, vanity::BytePattern, xoroshiro::BatchSeed,
    };
    use core::hint::black_box;

    fn batch(seed: u64, thread_idx: usize) -> (BatchSeed, u64) {
        let batch = BatchSeed {
            seed: seed.wrapping_sub(1),
            width: 32,
        };
        (black_box(batch), black_box(32 + thread_idx as u64))
    }

    fn error(result: CandidateResult) -> bool {
        result.status == CandidateResult::STATUS_ERROR && result.bytes == [0; 256]
    }

    /// The private key is the payload and the rest of the record is zero.
    pub fn matches(
        candidate: impl Fn(&BatchSeed, u64, &BytePattern) -> CandidateResult,
        seed: u64,
        thread_idx: usize,
        prefix: &[u8],
        suffix: &[u8],
        private_key: &[u8; 32],
    ) -> bool {
        let (batch, counter) = batch(seed, thread_idx);
        let pattern = black_box(BytePattern::new(prefix, suffix).unwrap());
        let result = candidate(&batch, counter, &pattern);
        let mut expected = [0u8; 256];
        expected[..32].copy_from_slice(private_key);
        result.status == CandidateResult::STATUS_MATCH && result.bytes == expected
    }

    pub fn misses(
        candidate: impl Fn(&BatchSeed, u64, &BytePattern) -> CandidateResult,
        seed: u64,
        thread_idx: usize,
        prefix: &[u8],
        suffix: &[u8],
    ) -> bool {
        let (batch, counter) = batch(seed, thread_idx);
        let pattern = black_box(BytePattern::new(prefix, suffix).unwrap());
        let result = candidate(&batch, counter, &pattern);
        result.status == CandidateResult::STATUS_MISS && result.bytes == [0; 256]
    }

    /// An empty batch and overlong patterns are errors with a zero record.
    pub fn rejects_invalid(
        candidate: impl Fn(&BatchSeed, u64, &BytePattern) -> CandidateResult,
        seed: u64,
    ) -> bool {
        let batch = black_box(BatchSeed { seed, width: 32 });
        let empty = black_box(BatchSeed { seed: 0, width: 0 });
        let mut pattern = BytePattern::new(b"", b"").unwrap();
        if !error(candidate(&empty, black_box(0), &black_box(pattern))) {
            return false;
        }
        pattern.prefix_len = 65;
        if !error(candidate(&batch, black_box(0), &black_box(pattern))) {
            return false;
        }
        pattern.prefix_len = 0;
        pattern.suffix_len = 65;
        error(candidate(&batch, black_box(0), &black_box(pattern)))
    }
}

#[cfg(any(
    feature = "self_test_p256_public_key",
    feature = "self_test_p256_signature"
))]
const P256_GENERATOR: [u8; 65] = [
    0x04, 0x6b, 0x17, 0xd1, 0xf2, 0xe1, 0x2c, 0x42, 0x47, 0xf8, 0xbc, 0xe6, 0xe5, 0x63, 0xa4, 0x40,
    0xf2, 0x77, 0x03, 0x7d, 0x81, 0x2d, 0xeb, 0x33, 0xa0, 0xf4, 0xa1, 0x39, 0x45, 0xd8, 0x98, 0xc2,
    0x96, 0x4f, 0xe3, 0x42, 0xe2, 0xfe, 0x1a, 0x7f, 0x9b, 0x8e, 0xe7, 0xeb, 0x4a, 0x7c, 0x0f, 0x9e,
    0x16, 0x2b, 0xce, 0x33, 0x57, 0x6b, 0x31, 0x5e, 0xce, 0xcb, 0xb6, 0x40, 0x68, 0x37, 0xbf, 0x51,
    0xf5,
];

/// sha256(b"sample")
#[cfg(any(feature = "self_test_p256_signature", feature = "self_test_rsa_pss"))]
const SAMPLE_SHA256: [u8; 32] = [
    0xaf, 0x2b, 0xdb, 0xe1, 0xaa, 0x9b, 0x6e, 0xc1, 0xe2, 0xad, 0xe1, 0xd6, 0x94, 0xf4, 0x1f, 0xc7,
    0x1a, 0x83, 0x1d, 0x02, 0x68, 0xe9, 0x89, 0x15, 0x62, 0x11, 0x3d, 0x8a, 0x62, 0xad, 0xd1, 0xbf,
];

/// Folds a candidate into the digest of an end-to-end check.
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
mod tests {
    use super::*;

    #[test]
    fn every_check_passes_on_cpu() {
        for mode in MODES {
            let mut results = alloc::vec![0xa5a5a5a5; mode.checks.len()];
            (mode.run)(&mut results);
            for (check, &result) in mode.checks.iter().zip(&results) {
                assert_eq!(
                    result, 1,
                    "{}.{} ({}) failed",
                    mode.name, check.name, check.label
                );
                assert!((check.run)(), "{}.{} failed", mode.name, check.name);
            }
        }
    }
}
