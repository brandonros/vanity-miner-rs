//! Shared RSA-PSS host/device launch contract.
use logic::{
    modes::rsa_pss::RsaPssRequest,
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
// SAFETY: repr(C), padding-free integer fields, all bit patterns valid.
unsafe impl logic::search::device_record::DeviceRecord for Launch {}

pub fn candidate(
    launch: &Launch,
    request: &RsaPssRequest,
    pattern: &HexPattern,
    message: &[u8],
    lane: u32,
) -> CandidateResult {
    if lane >= launch.count || launch.message_len != message.len() as u64 {
        return CandidateResult::ERROR;
    }
    match launch.start.checked_add(u64::from(lane)) {
        Some(counter) => logic::modes::rsa_pss::rsa_pss(request, message, counter, pattern),
        None => CandidateResult::ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layout_is_padding_free() {
        assert_eq!(core::mem::size_of::<Launch>(), 24);
        assert_eq!(core::mem::offset_of!(Launch, message_len), 8);
        assert_eq!(core::mem::offset_of!(Launch, count), 16);
        assert_eq!(core::mem::size_of::<RsaPssRequest>(), 920);
        assert_eq!(core::mem::size_of::<HexPattern>(), 516);
    }
}
