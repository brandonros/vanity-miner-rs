use seq_macro::seq;

const BASE58_ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const NUM_CHUNKS: usize = 8;
const MAX_REQUIRED_LIMBS: usize = 10;
const DIGITS_PER_LIMB: usize = 5; // log_58(2^32) ≈ 5.462
const NEXT_LIMB_DIVISOR: u64 = 58_u64.pow(DIGITS_PER_LIMB as u32); // 58^5 = 656,356,768
const DIVISORS: [u64; DIGITS_PER_LIMB] = {
    let mut divs = [0u64; DIGITS_PER_LIMB];
    let mut val = 1u64;
    let mut i = 0;
    while i < DIGITS_PER_LIMB {
        divs[i] = val;
        val *= 58;
        i += 1;
    }
    divs
};

pub fn base58_encode_32(input: &[u8; 32], output: &mut [u8; 64]) -> usize {
    // Count leading zeros in advance
    let mut num_leading_zeros = 0;
    for &byte in input.iter() {
        if byte == 0 {
            num_leading_zeros += 1;
        } else {
            break;
        }
    }

    // Chunk input into u32
    let chunks: [u32; NUM_CHUNKS] = [
        u32::from_be_bytes([input[0], input[1], input[2], input[3]]),
        u32::from_be_bytes([input[4], input[5], input[6], input[7]]),
        u32::from_be_bytes([input[8], input[9], input[10], input[11]]),
        u32::from_be_bytes([input[12], input[13], input[14], input[15]]),
        u32::from_be_bytes([input[16], input[17], input[18], input[19]]),
        u32::from_be_bytes([input[20], input[21], input[22], input[23]]),
        u32::from_be_bytes([input[24], input[25], input[26], input[27]]),
        u32::from_be_bytes([input[28], input[29], input[30], input[31]]),
    ];

    // Process input in chunks of bytes
    let mut limbs = [0u32; MAX_REQUIRED_LIMBS];
    let mut limb_count = 0;
    seq!(I in 0..8 {
        // Convert chunk to single value using bit manipulation
        let chunk = chunks[I];
        let carry = chunk as u64;
        let mut remaining_carry = carry;

        // Update existing limbs
        for i in 0..limb_count {
            remaining_carry += (limbs[i] as u64) << 32;
            limbs[i] = (remaining_carry % NEXT_LIMB_DIVISOR) as u32;
            remaining_carry = (remaining_carry / NEXT_LIMB_DIVISOR) as u64;
        }

        // Add new limbs
        if remaining_carry > 0 && limb_count < MAX_REQUIRED_LIMBS {
            limbs[limb_count] = (remaining_carry % NEXT_LIMB_DIVISOR) as u32;
            remaining_carry = (remaining_carry / NEXT_LIMB_DIVISOR) as u64;
            limb_count += 1;
            
            if remaining_carry > 0 && limb_count < MAX_REQUIRED_LIMBS {
                limbs[limb_count] = remaining_carry as u32;
                limb_count += 1;
            }
        }
    });

    // Convert limbs to bytes in Base58 format
    for idx in (0..limb_count).rev() {
        let limb_value = limbs[idx] as u64;
        let output_offset = idx * DIGITS_PER_LIMB;

        // Extract Base58 digits using precomputed divisors
        for i in 0..DIGITS_PER_LIMB {
            let temp = (limb_value / DIVISORS[i]) % 58;
            output[output_offset + i] = temp as u8;
        }
    }

    // Scale for remainder and apply alphabet
    let mut result_len = limb_count * DIGITS_PER_LIMB;

    // Trim leading zeros in the result
    while result_len > 0 && output[result_len - 1] == 0 {
        result_len -= 1;
    }

    // Add a zero byte for each leading zero in the input
    for _ in 0..num_leading_zeros {
        output[result_len] = 0;
        result_len += 1;
    }

    // Apply alphabet encoding
    for val in &mut output[..result_len] {
        *val = BASE58_ALPHABET[*val as usize];
    }

    // Reverse the result
    output[..result_len].reverse();

    result_len
}

pub fn base58_encode_25(input: &[u8; 25], output: &mut [u8; 64]) -> usize {
    const MAX_REQUIRED_LIMBS_25: usize = 8; // Sufficient for 25 bytes

    // Count leading zeros in advance
    let mut num_leading_zeros = 0;
    for &byte in input.iter() {
        if byte == 0 {
            num_leading_zeros += 1;
        } else {
            break;
        }
    }

    // Process input bytes through big integer arithmetic
    // We'll process all 25 bytes by treating the input as a big-endian number

    // Process input in chunks of bytes
    let mut limbs = [0u32; MAX_REQUIRED_LIMBS_25];
    let mut limb_count = 0;
    
    // Process each byte individually to avoid padding issues
    for &byte in input.iter() {
        let carry = byte as u64;
        let mut remaining_carry = carry;

        // Update existing limbs (multiply by 256 and add new byte)
        for i in 0..limb_count {
            remaining_carry += (limbs[i] as u64) << 8;
            limbs[i] = (remaining_carry % NEXT_LIMB_DIVISOR) as u32;
            remaining_carry = remaining_carry / NEXT_LIMB_DIVISOR;
        }

        // Add new limbs if needed
        while remaining_carry > 0 && limb_count < MAX_REQUIRED_LIMBS_25 {
            limbs[limb_count] = (remaining_carry % NEXT_LIMB_DIVISOR) as u32;
            remaining_carry = remaining_carry / NEXT_LIMB_DIVISOR;
            limb_count += 1;
        }
    }

    // Convert limbs to bytes in Base58 format
    for idx in (0..limb_count).rev() {
        let limb_value = limbs[idx] as u64;
        let output_offset = idx * DIGITS_PER_LIMB;

        // Extract Base58 digits using precomputed divisors
        for i in 0..DIGITS_PER_LIMB {
            let temp = (limb_value / DIVISORS[i]) % 58;
            output[output_offset + i] = temp as u8;
        }
    }

    // Scale for remainder and apply alphabet
    let mut result_len = limb_count * DIGITS_PER_LIMB;

    // Trim leading zeros in the result
    while result_len > 0 && output[result_len - 1] == 0 {
        result_len -= 1;
    }

    // Add a zero byte for each leading zero in the input
    for _ in 0..num_leading_zeros {
        output[result_len] = 0;
        result_len += 1;
    }

    // Apply alphabet encoding
    for val in &mut output[..result_len] {
        *val = BASE58_ALPHABET[*val as usize];
    }

    // Reverse the result
    output[..result_len].reverse();

    result_len
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_encode_32_correctly() {
        let public_key_bytes: [u8; 32] = hex::decode("0af764c1b6133a3a0abd7ef9c853791b687ce1e235f9dc8466d886da314dbea7").unwrap().try_into().unwrap();
        let mut bs58_encoded_public_key = [0u8; 64];
        let encoded_len = base58_encode_32(&public_key_bytes, &mut bs58_encoded_public_key);
        let bs58_encoded_public_key = &bs58_encoded_public_key[0..encoded_len];
        let expected = hex::decode("6a6f7365413875757746426a58707558423879453233437845756d596758336a486251677753627166504c").unwrap();
        assert_eq!(*bs58_encoded_public_key, *expected);
    }

    #[test]
    fn should_encode_25_correctly() {
        let public_key_bytes: [u8; 25] = hex::decode("0AF764C1B6133A3A0ABD7EF9C853791B687CE1E235F9DC8466").unwrap().try_into().unwrap();
        let mut bs58_encoded_public_key = [0u8; 64];
        let encoded_len = base58_encode_25(&public_key_bytes, &mut bs58_encoded_public_key);
        let bs58_encoded_public_key = &bs58_encoded_public_key[0..encoded_len];
        let expected = hex::decode("355177385441616239385172516D796D637A7A78776B5A7A61634D444C344D654548").unwrap();
        assert_eq!(*bs58_encoded_public_key, *expected);
    }
}
