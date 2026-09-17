//! Shared Bitcoin host/device launch contract.
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
        Some(counter) => logic::modes::bitcoin::candidate(&launch.seed, counter, &launch.pattern),
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
                seed: 10088153575472065218,
                width: 32,
            },
            start: 0,
            count: 32,
            audit: 1,
            pattern: BytePattern::new(b"bc1q", b"6m").unwrap(),
        };
        let result = candidate(&launch, 0);
        assert_eq!(result.status, CandidateResult::STATUS_MATCH);
        assert_eq!(
            &result.bytes[..32],
            &[
                0x23, 0xa3, 0x3f, 0x35, 0x73, 0x7a, 0xb1, 0xab, 0xc1, 0x6c, 0xc1, 0xd1, 0x75, 0x55,
                0xc8, 0xdc, 0x75, 0x18, 0x33, 0xac, 0x76, 0xcf, 0x4b, 0xc9, 0xe3, 0x2f, 0xaf, 0x3d,
                0x73, 0x52, 0xe9, 0x30,
            ]
        );
        launch.pattern.suffix[0] ^= 1;
        assert_eq!(candidate(&launch, 0).status, CandidateResult::STATUS_MISS);
        launch.pattern.prefix_len = 65;
        assert_eq!(candidate(&launch, 0).status, CandidateResult::STATUS_ERROR);
        launch.pattern = BytePattern::new(&[], &[]).unwrap();
        launch.seed.width = 0;
        assert_eq!(candidate(&launch, 0).status, CandidateResult::STATUS_ERROR);
        launch.seed.width = 32;
        launch.start = u64::MAX;
        assert_eq!(candidate(&launch, 1).status, CandidateResult::STATUS_ERROR);
    }
}
