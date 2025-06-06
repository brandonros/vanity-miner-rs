use crate::sha256;

pub fn shallenge(username: &[u8], username_len: usize, nonce: &[u8], nonce_len: usize) -> [u8; 32] {
    let mut input = [0u8; 32];
    let mut pos = 0;
    
    // Copy username
    input[pos..pos + username_len].copy_from_slice(&username[..username_len]);
    pos += username_len;
    
    // Add separator '/'
    input[pos] = b'/';
    pos += 1;
    
    // Copy nonce
    input[pos..pos + nonce_len].copy_from_slice(&nonce[..nonce_len]);
    pos += nonce_len;
    
    // Hash only the used portion
    sha256::sha256_32bytes_from_bytes(&input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shallenge() {
        // Test with 32 zero bytes
        let username: [u8; 10] = "brandonros".as_bytes().try_into().unwrap();
        let nonce: [u8; 21] = "000000000000000000000".as_bytes().try_into().unwrap();
        let result = shallenge(&username, 10, &nonce, 21);
        let expected: [u8; 32] = hex::decode("f7a41dae1196282f0a544a8c7f1bbf61bda79307dc424c0d9febd27b08e1bf78").unwrap().try_into().unwrap();
        assert_eq!(result, expected);
    }
}
