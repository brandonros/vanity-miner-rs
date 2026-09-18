//! Shared stock-Rust host/device request contract.
use logic::{
    modes::p256_signature::P256SignatureRequest,
    search::{candidate_result::CandidateResult, hex_pattern::HexPattern},
};
pub const INTERFACE: &str = include_str!("../kernel.interface.json");
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Launch {
    pub start: u64,
    pub message_len: u64,
    pub count: u32,
    pub audit: u32,
}
// SAFETY: repr(C), padding-free integer fields; all bit patterns valid.
unsafe impl logic::search::device_record::DeviceRecord for Launch {}
pub fn candidate(
    launch: &Launch,
    request: &P256SignatureRequest,
    pattern: &HexPattern,
    message: &[u8],
    lane: u32,
) -> CandidateResult {
    match launch.start.checked_add(u64::from(lane)) {
        Some(counter) => {
            logic::modes::p256_signature::p256_signature(request, message, counter, pattern)
        }
        None => CandidateResult::ERROR,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout() {
        assert_eq!(core::mem::size_of::<Launch>(), 24);
        assert_eq!(core::mem::offset_of!(Launch, message_len), 8);
        assert_eq!(core::mem::offset_of!(Launch, count), 16);
        assert_eq!(core::mem::size_of::<P256SignatureRequest>(), 168);
        assert_eq!(core::mem::size_of::<HexPattern>(), 516);
    }
}
