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
        assert_eq!(core::mem::size_of::<super::BatchResult>(), 272);
        assert_eq!(core::mem::size_of::<crate::hex_pattern::HexPattern>(), 516);
    }
}

/// One shared winner per synchronized launch, matching the original kernels.
/// Every matching lane increments `matches`; only the first writes the payload.
/// Errors are counted separately so a winning lane cannot hide a device failure.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BatchResult {
    pub matches: u32,
    pub errors: u32,
    pub lane: u32,
    pub candidate: CandidateResult,
}
impl BatchResult {
    pub const EMPTY: Self = Self {
        matches: 0,
        errors: 0,
        lane: u32::MAX,
        candidate: CandidateResult::MISS,
    };

    /// Validate synchronized output before using a device-supplied lane index.
    pub fn winner(&self, count: u32) -> Result<Option<(u32, CandidateResult)>, &'static str> {
        if count == 0 || self.matches > count || self.errors > count - self.matches {
            return Err("device returned invalid batch counts");
        }
        if self.errors != 0 {
            return Err("device candidate evaluation failed");
        }
        if self.matches == 0 {
            return Ok(None);
        }
        if self.lane >= count || self.candidate.status != 1 {
            return Err("device returned an invalid winner");
        }
        Ok(Some((self.lane, self.candidate)))
    }
}
impl zeroize::Zeroize for BatchResult {
    fn zeroize(&mut self) {
        self.matches.zeroize();
        self.errors.zeroize();
        self.lane.zeroize();
        self.candidate.zeroize();
    }
}

#[cfg(test)]
mod winner_tests {
    use super::*;

    #[test]
    fn shared_winner_may_be_any_matching_lane() {
        let output = BatchResult {
            matches: 3,
            errors: 0,
            lane: 63,
            candidate: CandidateResult::matched(&[42]),
        };
        let (lane, candidate) = output.winner(64).unwrap().unwrap();
        assert_eq!(lane, 63);
        assert_eq!(candidate.bytes[0], 42);
        assert!(BatchResult::EMPTY.winner(64).unwrap().is_none());
    }

    #[test]
    fn malformed_counts_and_winners_are_rejected() {
        let valid = BatchResult {
            matches: 1,
            errors: 0,
            lane: 0,
            candidate: CandidateResult::matched(&[42]),
        };
        for invalid in [
            BatchResult {
                matches: 65,
                ..valid
            },
            BatchResult { lane: 64, ..valid },
            BatchResult {
                candidate: CandidateResult::MISS,
                ..valid
            },
            BatchResult {
                matches: u32::MAX,
                errors: u32::MAX,
                ..valid
            },
        ] {
            assert!(invalid.winner(64).is_err());
        }
        assert!(valid.winner(0).is_err());
    }

    #[test]
    fn error_in_another_lane_is_not_hidden_by_winner() {
        let output = BatchResult {
            matches: 1,
            errors: 1,
            lane: 0,
            candidate: CandidateResult::matched(&[42]),
        };
        assert_eq!(
            output.winner(64).err(),
            Some("device candidate evaluation failed")
        );
    }
}
