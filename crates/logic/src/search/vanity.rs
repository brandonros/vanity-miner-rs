/// Check if data matches the given prefix and suffix patterns.
/// Works with any byte slice - encoded addresses (Solana/Bitcoin) or raw bytes (Ethereum).
pub fn check_vanity_match(data: &[u8], prefix: &[u8], suffix: &[u8]) -> bool {
    if prefix.len() > data.len() || suffix.len() > data.len() {
        return false;
    }
    for (actual, expected) in data.iter().zip(prefix) {
        if actual != expected {
            return false;
        }
    }

    for (actual, expected) in data.iter().rev().zip(suffix.iter().rev()) {
        if actual != expected {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn all_pattern_lengths_and_overlaps_match_slice_reference() {
        let data: [u8; 20] = core::array::from_fn(|i| i as u8);
        for n in 0..=64 {
            for m in 0..=64 {
                let mut prefix = [0u8; 64];
                let mut suffix = [0u8; 64];
                prefix[..n.min(20)].copy_from_slice(&data[..n.min(20)]);
                suffix[..m.min(20)].copy_from_slice(&data[20 - m.min(20)..]);
                for mismatch in [false, true] {
                    if mismatch {
                        if n > 0 {
                            prefix[n - 1] ^= 1;
                        }
                        if m > 0 {
                            suffix[0] ^= 1;
                        }
                    }
                    assert_eq!(
                        check_vanity_match(&data, &prefix[..n], &suffix[..m]),
                        data.starts_with(&prefix[..n]) && data.ends_with(&suffix[..m])
                    );
                }
            }
        }
        assert!(check_vanity_match(&[], &[], &[]));
        assert!(!check_vanity_match(&[], &[0], &[]));
    }

    #[test]
    fn test_prefix_match() {
        assert!(check_vanity_match(b"hello_world", b"hello", b""));
        assert!(!check_vanity_match(b"hello_world", b"world", b""));
    }

    #[test]
    fn test_suffix_match() {
        assert!(check_vanity_match(b"hello_world", b"", b"world"));
        assert!(!check_vanity_match(b"hello_world", b"", b"hello"));
    }

    #[test]
    fn test_prefix_and_suffix_match() {
        assert!(check_vanity_match(b"hello_world", b"hello", b"world"));
        assert!(!check_vanity_match(b"hello_world", b"hello", b"hello"));
    }

    #[test]
    fn test_empty_patterns() {
        assert!(check_vanity_match(b"anything", b"", b""));
    }

    #[test]
    fn test_pattern_longer_than_data() {
        assert!(!check_vanity_match(b"hi", b"hello", b""));
        assert!(!check_vanity_match(b"hi", b"", b"world"));
    }
}

/// Prefix and suffix bytes passed to an address-search kernel.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BytePattern {
    pub prefix_len: u32,
    pub suffix_len: u32,
    pub prefix: [u8; 64],
    pub suffix: [u8; 64],
}
impl BytePattern {
    pub fn new(prefix: &[u8], suffix: &[u8]) -> Result<Self, &'static str> {
        if prefix.len() > 64 || suffix.len() > 64 {
            return Err("address pattern exceeds 64 bytes");
        }
        let mut pattern = Self {
            prefix_len: prefix.len() as u32,
            suffix_len: suffix.len() as u32,
            prefix: [0; 64],
            suffix: [0; 64],
        };
        pattern.prefix[..prefix.len()].copy_from_slice(prefix);
        pattern.suffix[..suffix.len()].copy_from_slice(suffix);
        Ok(pattern)
    }
    pub fn parts(&self) -> Option<(&[u8], &[u8])> {
        Some((
            self.prefix.get(..self.prefix_len as usize)?,
            self.suffix.get(..self.suffix_len as usize)?,
        ))
    }
}
// SAFETY: repr(C), two u32 lengths and byte arrays; no padding or invalid bit patterns.
unsafe impl super::device_record::DeviceRecord for BytePattern {}
