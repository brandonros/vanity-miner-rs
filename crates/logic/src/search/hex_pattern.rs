//! Fixed-width hexadecimal matching shared by host and device searches.

/// Largest supported target: an RSA-2048 modulus or signature.
pub const MAX_HEX_TARGET_BYTES: usize = 256;

/// A byte mask avoids allocating or parsing strings inside device kernels.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HexPattern {
    value: [u8; MAX_HEX_TARGET_BYTES],
    mask: [u8; MAX_HEX_TARGET_BYTES],
    len: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatternError {
    InvalidTargetLength,
    InvalidHex,
    TooLong,
    Contradiction,
}

impl core::fmt::Display for PatternError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidTargetLength => "target width must be between 1 and 256 bytes",
            Self::InvalidHex => "patterns must contain only hexadecimal digits without 0x",
            Self::TooLong => "pattern exceeds the target width",
            Self::Contradiction => "pattern constraints contradict each other or the target format",
        })
    }
}

impl core::error::Error for PatternError {}

impl HexPattern {
    /// Lexicographically smallest matching byte string (free bits are zero).
    /// Useful for rejecting targets outside a key-dependent numeric range.
    pub fn minimum_value(&self) -> &[u8] {
        &self.value[..self.len as usize]
    }

    pub fn new(prefix: &str, suffix: &str, len: usize) -> Result<Self, PatternError> {
        if len == 0 || len > MAX_HEX_TARGET_BYTES {
            return Err(PatternError::InvalidTargetLength);
        }
        let mut pattern = Self {
            value: [0; MAX_HEX_TARGET_BYTES],
            mask: [0; MAX_HEX_TARGET_BYTES],
            len: len as u32,
        };
        if prefix.len() > len * 2 || suffix.len() > len * 2 {
            return Err(PatternError::TooLong);
        }
        for (text, start) in [(prefix, 0), (suffix, len * 2 - suffix.len())] {
            for (offset, c) in text.bytes().enumerate() {
                let digit = match c {
                    b'0'..=b'9' => c - b'0',
                    b'a'..=b'f' => c - b'a' + 10,
                    b'A'..=b'F' => c - b'A' + 10,
                    _ => return Err(PatternError::InvalidHex),
                };
                let nibble = start + offset;
                let shift = if nibble % 2 == 0 { 4 } else { 0 };
                pattern.constrain_byte(nibble / 2, 15 << shift, digit << shift)?;
            }
        }
        Ok(pattern)
    }

    /// Add known format bits, rejecting incompatible requested patterns.
    pub fn constrain_byte(
        &mut self,
        offset: usize,
        mask: u8,
        value: u8,
    ) -> Result<(), PatternError> {
        if offset >= self.len as usize {
            return Err(PatternError::InvalidTargetLength);
        }
        if (self.value[offset] ^ value) & self.mask[offset] & mask != 0 {
            return Err(PatternError::Contradiction);
        }
        self.value[offset] = (self.value[offset] & !mask) | (value & mask);
        self.mask[offset] |= mask;
        Ok(())
    }

    pub fn matches(&self, bytes: &[u8]) -> bool {
        bytes.len() == self.len as usize
            && bytes
                .iter()
                .enumerate()
                .all(|(i, byte)| (byte ^ self.value[i]) & self.mask[i] == 0)
    }

    /// Unique constrained bits, counting prefix/suffix overlap only once.
    /// Subtract known structural bits after adding format constraints to obtain
    /// the generic random-candidate work exponent. Constructive RSA search has
    /// a different cost and must not report this as its measured candidate rate.
    pub fn constrained_bits(&self) -> u32 {
        self.mask[..self.len as usize]
            .iter()
            .map(|byte| byte.count_ones())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_nibbles_case_and_exact_width() {
        let p = HexPattern::new("AbC", "dEf", 4).unwrap();
        assert!(p.matches(&[0xab, 0xc0, 0x0d, 0xef]));
        assert!(!p.matches(&[0xab, 0x0c, 0x0d, 0xef]));
        assert!(!p.matches(&[0xab, 0xc0, 0x0d, 0xef, 0]));
        assert_eq!(p.constrained_bits(), 24);
    }

    #[test]
    fn overlaps_are_checked_and_counted_once() {
        let p = HexPattern::new("abc", "bcd", 2).unwrap();
        assert!(p.matches(&[0xab, 0xcd]));
        assert_eq!(p.constrained_bits(), 16);
        assert!(matches!(
            HexPattern::new("abc", "ccd", 2),
            Err(PatternError::Contradiction)
        ));
    }

    #[test]
    fn malformed_and_excessive_patterns() {
        for text in ["0x", "zz", "é", " a", "a\n"] {
            assert!(HexPattern::new(text, "", 32).is_err());
        }
        assert!(matches!(
            HexPattern::new("aaa", "", 1),
            Err(PatternError::TooLong)
        ));
        assert!(HexPattern::new("", "", 0).is_err());
        assert!(HexPattern::new("", "", 257).is_err());
        assert!(HexPattern::new("", "", 1).unwrap().matches(&[0]));
    }

    #[test]
    fn rsa_structural_constraints() {
        let mut p = HexPattern::new("d", "f", 256).unwrap();
        p.constrain_byte(0, 0x80, 0x80).unwrap();
        p.constrain_byte(255, 1, 1).unwrap();
        assert_eq!(p.constrained_bits() - 2, 6);
        let mut low = HexPattern::new("7", "", 256).unwrap();
        assert_eq!(
            low.constrain_byte(0, 0x80, 0x80),
            Err(PatternError::Contradiction)
        );
        let mut even = HexPattern::new("", "e", 256).unwrap();
        assert_eq!(
            even.constrain_byte(255, 1, 1),
            Err(PatternError::Contradiction)
        );
    }

    #[test]
    fn sec1_fixed_tag_constraints() {
        let mut p = HexPattern::new("04a", "", 65).unwrap();
        p.constrain_byte(0, 0xff, 4).unwrap();
        assert_eq!(p.constrained_bits() - 8, 4);
        let mut wrong = HexPattern::new("05", "", 65).unwrap();
        assert_eq!(
            wrong.constrain_byte(0, 0xff, 4),
            Err(PatternError::Contradiction)
        );
    }
}

// SAFETY: repr(C) record containing only padding-free u32 fields and byte arrays.
unsafe impl super::device_record::DeviceRecord for HexPattern {}
