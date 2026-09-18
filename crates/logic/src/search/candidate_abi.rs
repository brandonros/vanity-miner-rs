//! Typed six-buffer candidate contract shared by host launchers and device entries.
//! Dynamic spans (payload and audit records) are bounded by Launch and the host.
use super::{candidate_result::CandidateResult, device_record::DeviceRecord};

llvm_metal_kernel::record! {
#[derive(Clone, Copy)]
pub struct Launch {
    pub start: u64,
    pub message_len: u64,
    pub count: u32,
    pub audit: u32,
}
}
impl Launch {
    pub fn counter(&self, lane: u32) -> Option<u64> {
        if lane >= self.count {
            None
        } else {
            self.start.checked_add(u64::from(lane))
        }
    }
}
// SAFETY: repr(C), padding-free integers, every bit pattern valid.
unsafe impl DeviceRecord for Launch {}

/// Semantic roles used by the search engine; the kernel declaration determines
/// their actual binding indices and serializes values in that generated order.
#[derive(Clone, Copy)]
pub struct Arguments<T> {
    pub launch: T,
    pub request: T,
    pub pattern: T,
    pub message: T,
    pub output: T,
    pub records: T,
}

/// A candidate entry with typed records, a payload and counter-based evaluation.
///
/// # Safety
/// The exported entry must implement the six-buffer layout emitted by descriptor(),
/// ignore padded lanes, preserve inputs, bound audit writes, and atomically claim
/// the winner. All record bit patterns must be handled safely. The host must
/// still validate dynamic lengths, disjoint storage, and counter ranges.
pub unsafe trait Contract {
    type Request: DeviceRecord;
    type Pattern: DeviceRecord;
    const ENTRY: &'static str;
    fn candidate(
        request: &Self::Request,
        pattern: &Self::Pattern,
        payload: &[u8],
        counter: u64,
    ) -> CandidateResult;
    fn validate_payload(_payload: &[u8]) -> Result<(), &'static str> {
        Ok(())
    }
    fn descriptor() -> &'static [u8];
    fn slots() -> Arguments<usize>;
    fn pack<T>(arguments: Arguments<T>) -> [T; 6];
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_grids_and_counter_exhaustion_never_wrap() {
        let launch = Launch {
            start: u64::MAX - 1,
            message_len: 0,
            count: 3,
            audit: 0,
        };
        assert_eq!(launch.counter(0), Some(u64::MAX - 1));
        assert_eq!(launch.counter(1), Some(u64::MAX));
        assert_eq!(launch.counter(2), None);
        assert_eq!(launch.counter(3), None);
        assert_eq!(Launch { count: 0, ..launch }.counter(0), None);
    }
}
