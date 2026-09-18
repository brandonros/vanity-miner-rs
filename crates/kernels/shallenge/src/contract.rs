//! Stock-Rust Shallenge entry and its shared host/device buffer contract.
use logic::search::{candidate_result::CandidateResult, xoroshiro::BatchSeed};

pub const INTERFACE: &str = include_str!("../kernel.interface.json");

/// Fixed-layout little-endian launch record; reserved bytes must be zero.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Launch {
    pub seed: BatchSeed,
    pub start: u64,
    pub count: u32,
    pub username_len: u32,
    pub target: [u8; 32],
    pub username: [u8; 32],
    pub audit: u32,
    pub reserved: u32,
}
// SAFETY: repr(C), no padding/pointers, every bit pattern valid. Sizes tested below.
unsafe impl logic::search::device_record::DeviceRecord for Launch {}

pub fn candidate(launch: &Launch, lane: u32) -> CandidateResult {
    if launch.username_len > 32 || launch.reserved != 0 {
        return CandidateResult::ERROR;
    }
    match launch.start.checked_add(u64::from(lane)) {
        Some(counter) => logic::modes::shallenge::candidate(
            &launch.seed,
            counter,
            &launch.target,
            &launch.username[..launch.username_len as usize],
        ),
        None => CandidateResult::ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn launch_layout_and_invalid_ranges() {
        assert_eq!(core::mem::size_of::<Launch>(), 104);
        assert_eq!(core::mem::offset_of!(Launch, target), 32);
        assert_eq!(core::mem::offset_of!(Launch, audit), 96);
        let mut launch = Launch {
            seed: BatchSeed {
                seed: 12345,
                width: 32,
            },
            start: 0,
            count: 32,
            username_len: 10,
            target: [255; 32],
            username: [b'a'; 32],
            audit: 1,
            reserved: 0,
        };
        assert_eq!(candidate(&launch, 0).status, CandidateResult::STATUS_MATCH);
        launch.start = u64::MAX;
        assert_eq!(candidate(&launch, 1).status, CandidateResult::STATUS_ERROR);
        launch.start = 0;
        launch.seed.width = 0;
        assert_eq!(candidate(&launch, 0).status, CandidateResult::STATUS_ERROR);
        launch.seed.width = 32;
        for length in [0, 31, 32, u32::MAX] {
            launch.username_len = length;
            assert_eq!(candidate(&launch, 0).status, CandidateResult::STATUS_ERROR);
        }
    }
}
