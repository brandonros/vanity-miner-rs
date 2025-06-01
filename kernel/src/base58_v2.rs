const BASE58_ALPHABET: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const INPUT_BYTES_PER_LIMB: usize = 4;
const DIGITS_PER_LIMB: usize = 5; // log_58(2^32) ≈ 5.462
const NEXT_LIMB_DIVISOR: u64 = 58_u64.pow(DIGITS_PER_LIMB as u32); 
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

/// Custom implementation to replace bytemuck::pod_align_to_mut
fn align_to_u32_mut(slice: &mut [u8]) -> (&mut [u8], &mut [u32], &mut [u8]) {
    // Calculate alignment requirements for u32
    const ALIGN: usize = core::mem::align_of::<u32>();
    const SIZE: usize = core::mem::size_of::<u32>();
    
    // Find the first index where we can align u32
    let mut start_idx = 0;
    while start_idx < slice.len() {
        let addr = slice.as_ptr() as usize + start_idx;
        if addr % ALIGN == 0 {
            break;
        }
        start_idx += 1;
    }
    
    // Calculate how many complete u32s we can fit
    let remaining = slice.len() - start_idx;
    let u32_count = remaining / SIZE;
    
    // Split the slice into three parts
    let (prefix, rest) = slice.split_at_mut(start_idx);
    let (u32_compatible, suffix) = rest.split_at_mut(u32_count * SIZE);
    
    // Get a properly-aligned view of the middle part as u32
    let u32_slice = unsafe {
        core::slice::from_raw_parts_mut(
            u32_compatible.as_mut_ptr() as *mut u32,
            u32_count
        )
    };
    
    (prefix, u32_slice, suffix)
}

pub fn base58_encode(input: &[u8], output: &mut [u8]) -> usize {
    // Get mutable references to different parts of the output buffer
    let (prefix, output_as_limbs, _) = align_to_u32_mut(output);
    let prefix_len = prefix.len();

    // Count leading zeros in advance
    let leading_zeros = input.iter().take_while(|&&b| b == 0).count();

    // Process input in chunks of bytes
    let mut index = 0;
    for chunk in input.chunks(INPUT_BYTES_PER_LIMB) {
        // Convert chunk to single value
        let mut carry: u64 = 0;
        let mut shift_size = 0;
        for input_byte in chunk {
            carry = (carry << 8) + *input_byte as u64;
            shift_size += 8;
        }

        // Update existing limbs
        for limb in &mut output_as_limbs[..index] {
            carry += (*limb as u64) << shift_size;
            *limb = (carry % NEXT_LIMB_DIVISOR) as u32;
            carry = carry / NEXT_LIMB_DIVISOR;
        }

        // Add new limbs as needed
        while carry > 0 {
            let limb = output_as_limbs.get_mut(index).unwrap();
            *limb = (carry % NEXT_LIMB_DIVISOR) as u32;
            carry = carry / NEXT_LIMB_DIVISOR;
            index += 1;
        }
    }

    // Convert limbs to bytes in Base58 format
    for idx in (0..index).rev() {
        let limb_offset = prefix_len + idx * INPUT_BYTES_PER_LIMB;
        let mut limb_bytes = [0; INPUT_BYTES_PER_LIMB];
        limb_bytes.copy_from_slice(&output[limb_offset..limb_offset + INPUT_BYTES_PER_LIMB]);

        let limb = if cfg!(target_endian = "little") {
            u32::from_le_bytes(limb_bytes)
        } else {
            u32::from_be_bytes(limb_bytes)
        };

        let output_offset = prefix_len + idx * DIGITS_PER_LIMB;
        let limb_value = limb as u64;

        // Extract Base58 digits more efficiently using precomputed divisors
        for i in 0..DIGITS_PER_LIMB {
            let temp = limb_value / DIVISORS[i];
            let temp = temp % 58;
            output[output_offset + i] = temp as u8;
        }
    }

    // Scale for remainder and apply alphabet
    let mut result_len = index * DIGITS_PER_LIMB;
    let output_slice = &mut output[prefix_len..];

    // Trim leading zeros in the result
    while result_len > 0 && output_slice[result_len - 1] == 0 {
        result_len -= 1;
    }

    // Add a zero byte for each leading zero in the input
    for _ in 0..leading_zeros {
        let byte = output_slice.get_mut(result_len).unwrap();
        *byte = 0;
        result_len += 1;
    }

    // Apply alphabet encoding
    for val in &mut output_slice[..result_len] {
        *val = BASE58_ALPHABET[*val as usize];
    }

    // Reverse the result
    output_slice[..result_len].reverse();

    // Move result to the beginning if there was a prefix
    if prefix_len > 0 {
        output.copy_within(prefix_len..prefix_len + result_len, 0);
    }

    result_len
}