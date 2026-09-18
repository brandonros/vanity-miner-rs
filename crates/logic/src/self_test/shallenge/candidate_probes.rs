//! Candidate probes owned by the shallenge self-test kernel.
use crate::search::candidate_result::CandidateResult;
use crate::self_test::black_box;

register_self_test! {
    /// shallenge candidate hash nonce length and padding
    fn candidate_match() -> u32 {
        use crate::search::xoroshiro::BatchSeed;
        let seed = black_box(BatchSeed {
            seed: 12313, // The second batch advances by 32 to the known-answer seed.
            width: 32,
        });
        let result = crate::modes::shallenge::candidate(
            &seed,
            black_box(32),
            &black_box([255; 32]),
            &black_box(*b"brandonros"),
        );
        let mut expected = [0u8; 256];
        expected[..32].copy_from_slice(&super::SHALLENGE_TEST_HASH);
        expected[32..53].copy_from_slice(&super::XOROSHIRO_NONCE_EXPECTED);
        expected[96] = 21; // u32 little-endian nonce length at byte offset 96.
        u32::from(result.status == CandidateResult::STATUS_MATCH && result.bytes == expected)
    }
}

register_self_test! {
    /// shallenge candidate equal and lower targets miss
    fn candidate_miss() -> u32 {
        use crate::search::xoroshiro::BatchSeed;
        let seed = black_box(BatchSeed {
            seed: 12313, // The second batch advances by 32 to the known-answer seed.
            width: 32,
        });
        for target in [super::SHALLENGE_TEST_HASH, [0; 32]] {
            let result = crate::modes::shallenge::candidate(
                &seed,
                black_box(32),
                &black_box(target),
                &black_box(*b"brandonros"),
            );
            if result.status != CandidateResult::STATUS_MISS || result.bytes != [0; 256] {
                return 0;
            }
        }
        1
    }
}

register_self_test! {
    /// shallenge candidate invalid username and seed
    fn candidate_invalid() -> u32 {
        use crate::search::xoroshiro::BatchSeed;
        let seed = black_box(BatchSeed {
            seed: 12345,
            width: 32,
        });
        let target = black_box([255; 32]);
        let result = crate::modes::shallenge::candidate(&seed, black_box(0), &target, &black_box(*b""));
        if result.status != CandidateResult::STATUS_ERROR || result.bytes != [0; 256] {
            return 0;
        }
        let result =
            crate::modes::shallenge::candidate(&seed, black_box(0), &target, &black_box([b'a'; 31]));
        if result.status != CandidateResult::STATUS_ERROR || result.bytes != [0; 256] {
            return 0;
        }
        let seed = black_box(BatchSeed {
            seed: 12345,
            width: 0,
        });
        let result =
            crate::modes::shallenge::candidate(&seed, black_box(0), &target, &black_box(*b"brandonros"));
        u32::from(result.status == CandidateResult::STATUS_ERROR && result.bytes == [0; 256])
    }
}
