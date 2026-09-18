//! Candidate probes owned by the bitcoin self-test kernel.
use crate::search::candidate_result::CandidateResult;
use core::hint::black_box;

register_self_test! {
    /// bitcoin candidate match payload
    fn candidate_match() -> u32 {
        use crate::search::{vanity::BytePattern, xoroshiro::BatchSeed};
        let seed = black_box(BatchSeed {
            seed: 13278869120712471060, // The second batch advances by 32 to the known-answer seed.
            width: 32,
        });
        let pattern = black_box(BytePattern::new(b"bc1q", b"ju9s4").unwrap());
        let result = crate::modes::bitcoin::candidate(&seed, black_box(33), &pattern);
        // Fixed expected payload, including the zero-filled remainder of the record.
        let mut expected = [0u8; 256];
        expected[..32].copy_from_slice(&super::BITCOIN_TEST_PRIV);
        u32::from(result.status == CandidateResult::STATUS_MATCH && result.bytes == expected)
    }
}

register_self_test! {
    /// bitcoin candidate suffix mismatch
    fn candidate_miss() -> u32 {
        use crate::search::{vanity::BytePattern, xoroshiro::BatchSeed};
        let seed = black_box(BatchSeed {
            seed: 13278869120712471060, // The second batch advances by 32 to the known-answer seed.
            width: 32,
        });
        let pattern = black_box(BytePattern::new(b"bc1q", b"ju9s5").unwrap());
        let result = crate::modes::bitcoin::candidate(&seed, black_box(33), &pattern);
        u32::from(result.status == CandidateResult::STATUS_MISS && result.bytes == [0; 256])
    }
}

register_self_test! {
    /// bitcoin candidate invalid seed and pattern
    fn candidate_invalid() -> u32 {
        use crate::search::{vanity::BytePattern, xoroshiro::BatchSeed};
        let seed = black_box(BatchSeed {
            seed: 13278869120712471092,
            width: 32,
        });
        let mut pattern = BytePattern::new(b"", b"").unwrap();
        let bad_seed = black_box(BatchSeed { seed: 0, width: 0 });
        let result = crate::modes::bitcoin::candidate(&bad_seed, black_box(0), &black_box(pattern));
        if result.status != CandidateResult::STATUS_ERROR || result.bytes != [0; 256] {
            return 0;
        }
        pattern.prefix_len = 65;
        let result = crate::modes::bitcoin::candidate(&seed, black_box(0), &black_box(pattern));
        if result.status != CandidateResult::STATUS_ERROR || result.bytes != [0; 256] {
            return 0;
        }
        pattern.prefix_len = 0;
        pattern.suffix_len = 65;
        let result = crate::modes::bitcoin::candidate(&seed, black_box(0), &black_box(pattern));
        u32::from(result.status == CandidateResult::STATUS_ERROR && result.bytes == [0; 256])
    }
}
