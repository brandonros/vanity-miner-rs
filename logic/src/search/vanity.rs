/// Check if data matches the given prefix and suffix patterns.
/// Works with any byte slice - encoded addresses (Solana/Bitcoin) or raw bytes (Ethereum).
pub fn check_vanity_match(data: &[u8], prefix: &[u8], suffix: &[u8]) -> bool {
    let len = data.len();

    // Check prefix
    if prefix.len() > len {
        return false;
    }
    for i in 0..prefix.len() {
        if data[i] != prefix[i] {
            return false;
        }
    }

    // Check suffix
    if suffix.len() > len {
        return false;
    }
    for i in 0..suffix.len() {
        if data[len - suffix.len() + i] != suffix[i] {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod test {
    use super::*;

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
