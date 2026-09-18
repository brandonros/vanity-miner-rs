//! Shared Solana host/device launch contract.
use logic::search::{candidate_result::CandidateResult, vanity::BytePattern, xoroshiro::BatchSeed};

pub const INTERFACE: &str = include_str!("../kernel.interface.json");

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Launch {
    pub seed: BatchSeed,
    pub start: u64,
    pub count: u32,
    pub audit: u32,
    pub pattern: BytePattern,
}
// SAFETY: repr(C), no implicit padding or pointers, all bit patterns valid.
unsafe impl logic::search::device_record::DeviceRecord for Launch {}

pub fn candidate(launch: &Launch, lane: u32) -> CandidateResult {
    match launch.start.checked_add(u64::from(lane)) {
        Some(counter) => logic::modes::solana::candidate(&launch.seed, counter, &launch.pattern),
        None => CandidateResult::ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matching_and_invalid_requests() {
        assert_eq!(core::mem::size_of::<Launch>(), 168);
        assert_eq!(core::mem::offset_of!(Launch, pattern), 32);
        let mut launch = Launch {
            seed: BatchSeed {
                seed: 583437459223573146,
                width: 32,
            },
            start: 0,
            count: 32,
            audit: 1,
            pattern: BytePattern::new(b"aaa", b"NFC").unwrap(),
        };
        let result = candidate(&launch, 3);
        assert_eq!(result.status, CandidateResult::STATUS_MATCH);
        assert_eq!(
            &result.bytes[..32],
            &[
                0xfa, 0x9c, 0xe9, 0xb0, 0x2d, 0xc2, 0x8a, 0x48, 0xf7, 0xe9, 0xd1, 0x55, 0x06, 0xd3,
                0xd2, 0xc4, 0x43, 0xd5, 0x96, 0x56, 0x5f, 0xa0, 0x52, 0x14, 0xb0, 0xff, 0x7c, 0x5a,
                0xb5, 0xe7, 0x95, 0x6b,
            ]
        );
        launch.pattern.suffix[0] ^= 1;
        assert_eq!(candidate(&launch, 3).status, CandidateResult::STATUS_MISS);
        launch.pattern.prefix_len = 65;
        assert_eq!(candidate(&launch, 3).status, CandidateResult::STATUS_ERROR);
        launch.pattern = BytePattern::new(&[], &[]).unwrap();
        launch.seed.width = 0;
        assert_eq!(candidate(&launch, 3).status, CandidateResult::STATUS_ERROR);
        launch.seed.width = 32;
        launch.start = u64::MAX;
        assert_eq!(candidate(&launch, 1).status, CandidateResult::STATUS_ERROR);
    }
}
