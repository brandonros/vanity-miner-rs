//! Shared fixed-layout result record for cryptographic candidate kernels.
//! Host and device use the same functions. Records containing secrets must be
//! cleared by their owners after the synchronized launch; never log them.

/// A lane returns only public output (or an RSA factor for host validation).
/// Status: 0 = miss, 1 = match, 2 = invalid request or failed arithmetic.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CandidateResult {
    pub status: u32,
    pub bytes: [u8; 256],
}
impl CandidateResult {
    pub const MISS: Self = Self {
        status: 0,
        bytes: [0; 256],
    };
    pub const ERROR: Self = Self {
        status: 2,
        bytes: [0; 256],
    };
    pub fn matched(bytes: &[u8]) -> Self {
        let mut result = Self::MISS;
        if bytes.len() > result.bytes.len() {
            return Self::ERROR;
        }
        result.status = 1;
        result.bytes[..bytes.len()].copy_from_slice(bytes);
        result
    }
}

impl zeroize::Zeroize for CandidateResult {
    fn zeroize(&mut self) {
        self.status.zeroize();
        self.bytes.zeroize();
    }
}

#[cfg(test)]
mod abi_tests {
    #[test]
    fn records_have_no_implicit_padding() {
        assert_eq!(core::mem::size_of::<super::CandidateResult>(), 260);
        assert_eq!(core::mem::size_of::<crate::hex_pattern::HexPattern>(), 516);
    }
}
