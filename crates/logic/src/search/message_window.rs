use crate::crypto::sha256::Sha256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowError {
    InvalidBounds,
    Exhausted,
}

impl core::fmt::Display for WindowError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidBounds => "mutable message window must be nonempty and within the message",
            Self::Exhausted => "candidate counter exceeds the message window capacity",
        })
    }
}

impl core::error::Error for WindowError {}

/// Writes an injective big-endian counter encoding exclusively inside a window.
/// Runners partition the global counter space, for example by assigning disjoint
/// batches. Never use per-worker counters without a globally unique mapping.
pub fn write_message_counter(
    message: &mut [u8],
    offset: usize,
    length: usize,
    counter: u128,
) -> Result<(), WindowError> {
    let end = offset
        .checked_add(length)
        .ok_or(WindowError::InvalidBounds)?;
    if length == 0 || end > message.len() {
        return Err(WindowError::InvalidBounds);
    }
    if length < 16 && counter >= (1u128 << (length * 8)) {
        return Err(WindowError::Exhausted);
    }
    let bytes = counter.to_be_bytes();
    let window = &mut message[offset..end];
    window.fill(0);
    for (destination, byte) in window.iter_mut().rev().zip(bytes.iter().rev()) {
        *destination = *byte;
    }
    Ok(())
}

/// Hash a virtual mutable window without allocating a message per device lane.
/// This is byte-for-byte equivalent to `write_message_counter` followed by SHA-256.
pub fn hash_message_counter(
    message: &[u8],
    offset: usize,
    length: usize,
    counter: u128,
) -> Result<[u8; 32], WindowError> {
    let end = offset
        .checked_add(length)
        .ok_or(WindowError::InvalidBounds)?;
    if length == 0 || end > message.len() {
        return Err(WindowError::InvalidBounds);
    }
    if length < 16 && counter >= (1u128 << (length * 8)) {
        return Err(WindowError::Exhausted);
    }
    let mut hash = Sha256::new();
    hash.update(&message[..offset]);
    let zeros = [0u8; 64];
    let mut padding = length.saturating_sub(16);
    while padding != 0 {
        let chunk = padding.min(zeros.len());
        hash.update(&zeros[..chunk]);
        padding -= chunk;
    }
    let bytes = counter.to_be_bytes();
    hash.update(&bytes[16 - length.min(16)..]);
    hash.update(&message[end..]);
    Ok(hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streaming_window_hash_equals_materialized_message() {
        for length in [1, 2, 8, 15, 16, 17, 63, 64, 65, 129] {
            for counter in [0, 1, 255] {
                let original = [0xa5; 160];
                let mut materialized = original;
                write_message_counter(&mut materialized, 7, length, counter).unwrap();
                let expected: [u8; 32] = Sha256::digest(materialized);
                assert_eq!(
                    hash_message_counter(&original, 7, length, counter).unwrap(),
                    expected
                );
            }
        }
        assert_eq!(
            hash_message_counter(&[], 0, 0, 0),
            Err(WindowError::InvalidBounds)
        );
        assert_eq!(
            hash_message_counter(&[0; 8], usize::MAX, 2, 0),
            Err(WindowError::InvalidBounds)
        );
        assert_eq!(
            hash_message_counter(&[0; 8], 0, 1, 256),
            Err(WindowError::Exhausted)
        );
    }

    #[test]
    fn window_uniqueness_and_boundaries() {
        let mut message = [0xa5; 6];
        for counter in 0..=255 {
            write_message_counter(&mut message, 2, 1, counter).unwrap();
            assert_eq!(message, [0xa5, 0xa5, counter as u8, 0xa5, 0xa5, 0xa5]);
        }
        let original = message;
        assert_eq!(
            write_message_counter(&mut message, 2, 1, 256),
            Err(WindowError::Exhausted)
        );
        assert_eq!(message, original);
        for (offset, length) in [(0, 0), (6, 1), (usize::MAX, 2), (2, usize::MAX)] {
            assert_eq!(
                write_message_counter(&mut message, offset, length, 0),
                Err(WindowError::InvalidBounds)
            );
            assert_eq!(message, original);
        }
    }

    #[test]
    fn large_window_and_counter_endianness() {
        let mut message = [0xa5; 34];
        write_message_counter(&mut message, 1, 32, 0x0102).unwrap();
        assert_eq!(message[0], 0xa5);
        assert_eq!(message[33], 0xa5);
        assert_eq!(&message[1..31], &[0; 30]);
        assert_eq!(&message[31..33], &[1, 2]);
        write_message_counter(&mut message, 1, 16, u128::MAX).unwrap();
        assert_eq!(&message[1..17], &[255; 16]);
    }
}
