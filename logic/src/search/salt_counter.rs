use super::message_window::WindowError;

/// Enumerate fixed-width salts by adding a unique counter to a random starting
/// value modulo 2^(8*length). The complete finite space can be visited without
/// repetition; a zero-byte salt has exactly one candidate.
pub fn write_salt_counter(base: &[u8], counter: u64, output: &mut [u8]) -> Result<(), WindowError> {
    if base.len() != output.len() {
        return Err(WindowError::InvalidBounds);
    }
    if base.len() < 8 && counter >= (1u64 << (base.len() * 8)) {
        return Err(WindowError::Exhausted);
    }
    let mut carry = counter as u128;
    for i in (0..base.len()).rev() {
        let sum = base[i] as u128 + (carry & 255);
        output[i] = sum as u8;
        carry = (carry >> 8) + (sum >> 8);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn salt_enumeration_wraps_without_repetition() {
        let mut seen = [false; 256];
        let mut output = [0];
        for counter in 0..256 {
            write_salt_counter(&[173], counter, &mut output).unwrap();
            assert!(!seen[output[0] as usize]);
            seen[output[0] as usize] = true;
        }
        assert_eq!(
            write_salt_counter(&[173], 256, &mut output),
            Err(WindowError::Exhausted)
        );
        let mut wide = [0; 16];
        write_salt_counter(&[255; 16], 1, &mut wide).unwrap();
        assert_eq!(wide, [0; 16]);
        write_salt_counter(&[], 0, &mut []).unwrap();
        assert_eq!(
            write_salt_counter(&[], 1, &mut []),
            Err(WindowError::Exhausted)
        );
    }
}
