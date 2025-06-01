use seq_macro::seq;

// Base58 alphabet (Bitcoin/IPFS standard)
const ALPHABET: &[u8; 58] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/// Maximum output length for 32-byte input
/// log(256^32) / log(58) ≈ 43.67, so 44 characters max
const MAX_OUTPUT_LEN: usize = 44;

// Use direct division instead of lookup tables for u32 limbs
// Good balance between performance and memory efficiency

/// Check if the big integer is zero (u32 version)
#[inline(always)]
fn is_zero(num: &[u32; 8]) -> bool {
    num.iter().all(|&x| x == 0)
}

#[inline(always)]
fn div_by_58(num: &mut [u32; 8]) -> u32 {
    let mut remainder = 0u64;
    
    // Process from most significant to least significant
    seq!(I in 0..8 {
        let temp = remainder * (1u64 << 32) + (num[I] as u64); // remainder * 2^32 + limb
        num[I] = (temp / 58) as u32;
        remainder = temp % 58;
    });
    
    remainder as u32
}

/// Convert 32 bytes to 8 u32 limbs (big-endian)
#[inline(always)]
fn bytes_to_u32_limbs(input: &[u8; 32]) -> [u32; 8] {
    let mut limbs = [0u32; 8];
    seq!(I in 0..8 {
        limbs[I] = u32::from_be_bytes([
            input[I * 4], input[I * 4 + 1], input[I * 4 + 2], input[I * 4 + 3]
        ]);
    });
    limbs
}

/// Version using u32 limbs (8 limbs for 32-byte input)
pub fn base58_encode(input: &[u8; 32], output: &mut [u8]) -> usize {
    // Count leading zeros for proper padding
    let num_leading_zeros = input.iter().take_while(|&&b| b == 0).count();
    
    // Convert input bytes to u32 limbs
    let mut num = bytes_to_u32_limbs(input);
    
    // Perform base conversion by repeated division
    let mut result_len = 0;
    let mut temp_output = [0u8; MAX_OUTPUT_LEN];
    
    while !is_zero(&num) {
        let remainder = div_by_58(&mut num);
        temp_output[result_len] = ALPHABET[remainder as usize];
        result_len += 1;
    }
    
    // Add leading '1's for leading zero bytes
    for i in 0..num_leading_zeros {
        output[i] = b'1';
    }
    
    // Reverse the digits
    for i in 0..result_len {
        output[num_leading_zeros + i] = temp_output[result_len - 1 - i];
    }
    
    num_leading_zeros + result_len
}
