use seq_macro::seq;

const BASE58_ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
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

pub fn base58_encode(input: &[u8; 32], output: &mut [u8]) -> usize {
    // Count leading zeros in advance
    let mut num_leading_zeros = 0;
    for &byte in input.iter() {
        if byte == 0 {
            num_leading_zeros += 1;
        } else {
            break;
        }
    }

    // Process input in chunks of bytes
    let mut limbs = [0u32; MAX_REQUIRED_LIMBS];
    let mut limb_count = 0;

    let chunks = [
        u32::from_be_bytes([input[0], input[1], input[2], input[3]]),
        u32::from_be_bytes([input[4], input[5], input[6], input[7]]),
        u32::from_be_bytes([input[8], input[9], input[10], input[11]]),
        u32::from_be_bytes([input[12], input[13], input[14], input[15]]),
        u32::from_be_bytes([input[16], input[17], input[18], input[19]]),
        u32::from_be_bytes([input[20], input[21], input[22], input[23]]),
        u32::from_be_bytes([input[24], input[25], input[26], input[27]]),
        u32::from_be_bytes([input[28], input[29], input[30], input[31]]),
    ];

    seq!(I in 0..8 {
        // Convert chunk to single value using bit manipulation
        let chunk = chunks[I];
        let carry = chunk as u64;
        let mut remaining_carry = carry;

        // Update existing limbs
        for i in 0..limb_count {
            remaining_carry += (limbs[i] as u64) << 32;
            limbs[i] = (remaining_carry % NEXT_LIMB_DIVISOR) as u32;
            remaining_carry /= NEXT_LIMB_DIVISOR;
        }

        // Add new limbs - unrolled for common cases
        if remaining_carry > 0 && limb_count < MAX_REQUIRED_LIMBS {
            limbs[limb_count] = (remaining_carry % NEXT_LIMB_DIVISOR) as u32;
            remaining_carry /= NEXT_LIMB_DIVISOR;
            limb_count += 1;
            
            if remaining_carry > 0 && limb_count < MAX_REQUIRED_LIMBS {
                limbs[limb_count] = remaining_carry as u32;
                limb_count += 1;
            }
        }
    });

    // Convert limbs to bytes in Base58 format
    let mut temp_output = [0u8; 64];
    
    for idx in (0..limb_count).rev() {
        let limb_value = limbs[idx] as u64;
        let output_offset = idx * DIGITS_PER_LIMB;

        // Extract Base58 digits using precomputed divisors
        for i in 0..DIGITS_PER_LIMB {
            let temp = limb_value / DIVISORS[i];
            let temp = temp % 58;
            temp_output[output_offset + i] = temp as u8;
        }
    }

    // Scale for remainder and apply alphabet
    let mut result_len = limb_count * DIGITS_PER_LIMB;

    // Trim leading zeros in the result
    while result_len > 0 && temp_output[result_len - 1] == 0 {
        result_len -= 1;
    }

    // Add a zero byte for each leading zero in the input
    for _ in 0..num_leading_zeros {
        temp_output[result_len] = 0;
        result_len += 1;
    }

    // Apply alphabet encoding
    for val in &mut temp_output[..result_len] {
        *val = BASE58_ALPHABET[*val as usize];
    }

    // Reverse the result
    temp_output[..result_len].reverse();

    // Copy to output
    output[..result_len].copy_from_slice(&temp_output[..result_len]);

    result_len
}
