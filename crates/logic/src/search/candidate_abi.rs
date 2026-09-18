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
