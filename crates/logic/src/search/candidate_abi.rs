//! Typed six-buffer candidate contract shared by host launchers and device entries.
//! Dynamic spans (payload and audit records) are bounded by Launch and the host.
use super::{
    candidate_result::{BatchResult, CandidateResult},
    device_record::DeviceRecord,
};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Launch {
    pub start: u64,
    pub message_len: u64,
    pub count: u32,
    pub audit: u32,
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

#[derive(Clone, Copy)]
pub struct Argument {
    pub name: &'static str,
    pub access: &'static str,
    pub bytes: usize,
    pub alignment: usize,
}
const fn record<T: DeviceRecord>(name: &'static str, access: &'static str) -> Argument {
    Argument {
        name,
        access,
        bytes: size_of::<T>(),
        alignment: align_of::<T>(),
    }
}

/// A candidate entry with typed records, a payload and counter-based evaluation.
///
/// # Safety
/// The exported entry must implement the six-buffer layout returned by layout(),
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
    fn layout() -> [Argument; 6] {
        [
            record::<Launch>("launch", "read"),
            record::<Self::Request>("request", "read"),
            record::<Self::Pattern>("pattern", "read"),
            record::<[u8; 1]>("message", "read"),
            record::<BatchResult>("output", "read_write"),
            record::<CandidateResult>("records", "write"),
        ]
    }
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
    #[test]
    fn launch_is_a_padding_free_portable_record() {
        assert_eq!(
            size_of::<Launch>(),
            2 * size_of::<u64>() + 2 * size_of::<u32>()
        );
        assert_eq!(core::mem::offset_of!(Launch, message_len), size_of::<u64>());
        assert_eq!(core::mem::offset_of!(Launch, count), 2 * size_of::<u64>());
        assert_eq!(
            core::mem::offset_of!(Launch, audit),
            2 * size_of::<u64>() + size_of::<u32>()
        );
    }
}
