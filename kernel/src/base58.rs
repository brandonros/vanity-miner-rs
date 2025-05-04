const BASE58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

pub fn encode(input: &[u8], output: &mut [u8]) -> usize {
    // Output buffer must be provided by the caller
    // The caller needs to ensure output has enough capacity
    
    // Count leading zeroes
    let mut total = 0;
    for i in 0..input.len() {
        if input[i] == 0 {
            output[total] = BASE58_ALPHABET[0];
            total += 1;
        } else {
            break;
        }
    }

    // Skip over the leading zeros in input
    let input_slice = if total < input.len() { &input[total..] } else { &[] };
    let in_len = input_slice.len();

    // encoding
    let mut idx = 0;
    for i in 0..in_len {
        let mut carry = input_slice[i] as u32;
        for j in 0..idx {
            carry += (output[total + j] as u32) << 8;
            output[total + j] = (carry % 58) as u8;
            carry /= 58;
        }
        while carry > 0 {
            output[total + idx] = (carry % 58) as u8;
            idx += 1;
            carry /= 58;
        }
    }

    // apply alphabet and reverse
    let c_idx = idx >> 1;
    for i in 0..c_idx {
        let temp = output[total + i];
        output[total + i] = output[total + idx - i - 1];
        output[total + idx - i - 1] = temp;
    }

    // Now apply the alphabet
    for i in 0..idx {
        output[total + i] = BASE58_ALPHABET[output[total + i] as usize];
    }

    // Return the actual size used
    total + idx
}
