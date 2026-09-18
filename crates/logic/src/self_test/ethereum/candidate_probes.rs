//! Candidate probes owned by the ethereum self-test kernel.
use crate::search::candidate_result::CandidateResult;
use crate::self_test::black_box;

register_self_test! {
    /// ethereum candidate match payload
    fn candidate_match() -> u32 {
        use crate::search::{vanity::BytePattern, xoroshiro::BatchSeed};
        let seed = black_box(BatchSeed {
            seed: 15455378110306975709, // The second batch advances by 32 to the known-answer seed.
            width: 32,
        });
        let Ok(pattern) = BytePattern::new(&[0x55, 0x55], &[0x65, 0x1c]) else { return 0; };
        let pattern = black_box(pattern);
        let result = crate::modes::ethereum::candidate(&seed, black_box(32), &pattern);
        // Fixed expected payload, including the zero-filled remainder of the record.
        let mut expected = [0u8; 256];
        expected[..32].copy_from_slice(&super::ETHEREUM_TEST_PRIV);
        u32::from(result.status == CandidateResult::STATUS_MATCH && result.bytes == expected)
    }
}

register_self_test! {
    /// ethereum candidate suffix mismatch
    fn candidate_miss() -> u32 {
        use crate::search::{vanity::BytePattern, xoroshiro::BatchSeed};
        let seed = black_box(BatchSeed {
            seed: 15455378110306975709, // The second batch advances by 32 to the known-answer seed.
            width: 32,
        });
        let Ok(pattern) = BytePattern::new(&[0x55, 0x55], &[0x65, 0x1d]) else { return 0; };
        let pattern = black_box(pattern);
        let result = crate::modes::ethereum::candidate(&seed, black_box(32), &pattern);
        u32::from(result.status == CandidateResult::STATUS_MISS && result.bytes == [0; 256])
    }
}

register_self_test! {
    /// ethereum candidate invalid seed and pattern
    fn candidate_invalid() -> u32 {
        use crate::search::{vanity::BytePattern, xoroshiro::BatchSeed};
        let seed = black_box(BatchSeed {
            seed: 15455378110306975741,
            width: 32,
        });
        let Ok(mut pattern) = BytePattern::new(b"", b"") else { return 0; };
        let bad_seed = black_box(BatchSeed { seed: 0, width: 0 });
        let result = crate::modes::ethereum::candidate(&bad_seed, black_box(0), &black_box(pattern));
        if result.status != CandidateResult::STATUS_ERROR || result.bytes != [0; 256] {
            return 0;
        }
        pattern.prefix_len = 65;
        let result = crate::modes::ethereum::candidate(&seed, black_box(0), &black_box(pattern));
        if result.status != CandidateResult::STATUS_ERROR || result.bytes != [0; 256] {
            return 0;
        }
        pattern.prefix_len = 0;
        pattern.suffix_len = 65;
        let result = crate::modes::ethereum::candidate(&seed, black_box(0), &black_box(pattern));
        u32::from(result.status == CandidateResult::STATUS_ERROR && result.bytes == [0; 256])
    }
}
