use seq_macro::seq;

// RIPEMD-160 initial hash values
const H0: [u32; 5] = [
    0x67452301,
    0xefcdab89,
    0x98badcfe,
    0x10325476,
    0xc3d2e1f0,
];

// Selection of message word for left line
const R_LEFT: [usize; 80] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13,
];

// Selection of message word for right line
const R_RIGHT: [usize; 80] = [
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11,
];

// Amount for left rotate for left line
const S_LEFT: [u32; 80] = [
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6,
];

// Amount for left rotate for right line
const S_RIGHT: [u32; 80] = [
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11,
];

// Constants for left line
const K_LEFT: [u32; 5] = [0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e];

// Constants for right line
const K_RIGHT: [u32; 5] = [0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000];

#[inline(always)]
const fn rotl32(x: u32, n: u32) -> u32 {
    (x << n) | (x >> (32 - n))
}

#[inline(always)]
const fn f(j: usize, x: u32, y: u32, z: u32) -> u32 {
    match j {
        0..=15 => x ^ y ^ z,
        16..=31 => (x & y) | (!x & z),
        32..=47 => (x | !y) ^ z,
        48..=63 => (x & z) | (y & !z),
        64..=79 => x ^ (y | !z),
        _ => unreachable!(),
    }
}

fn ripemd160_32bytes_u32(input: [u32; 8]) -> [u32; 5] {
    // Message schedule array - we need 16 words for RIPEMD-160 (one 512-bit block)
    let mut x = [0u32; 16];
    
    // Copy input to first 8 words (little-endian for RIPEMD-160)
    seq!(I in 0..8 {
        x[I] = input[I].to_le();
    });
    
    // Add padding: single 1 bit followed by zeros, then length
    // For 32 bytes (256 bits), padding is: 0x00000080, then zeros, then length = 256
    x[8] = 0x00000080;
    // x[9] through x[13] are already zero
    x[14] = 256; // Lower 32 bits of length (32 bytes = 256 bits)
    x[15] = 0;   // Upper 32 bits of length (always 0 for our use case)
    
    // Initialize working variables
    let mut al = H0[0];
    let mut bl = H0[1];
    let mut cl = H0[2];
    let mut dl = H0[3];
    let mut el = H0[4];
    
    let mut ar = H0[0];
    let mut br = H0[1];
    let mut cr = H0[2];
    let mut dr = H0[3];
    let mut er = H0[4];
    
    // Main loop - 80 rounds for each line
    seq!(J in 0..80 {
        // Left line
        let t = al
            .wrapping_add(f(J, bl, cl, dl))
            .wrapping_add(x[R_LEFT[J]])
            .wrapping_add(K_LEFT[J / 16]);
        let t = rotl32(t, S_LEFT[J]).wrapping_add(el);
        
        al = el;
        el = dl;
        dl = rotl32(cl, 10);
        cl = bl;
        bl = t;
        
        // Right line
        let t = ar
            .wrapping_add(f(79 - J, br, cr, dr))
            .wrapping_add(x[R_RIGHT[J]])
            .wrapping_add(K_RIGHT[J / 16]);
        let t = rotl32(t, S_RIGHT[J]).wrapping_add(er);
        
        ar = er;
        er = dr;
        dr = rotl32(cr, 10);
        cr = br;
        br = t;
    });
    
    // Combine results
    [
        H0[1].wrapping_add(cl).wrapping_add(dr),  // t
        H0[2].wrapping_add(dl).wrapping_add(er),
        H0[3].wrapping_add(el).wrapping_add(ar), 
        H0[4].wrapping_add(al).wrapping_add(br),
        H0[0].wrapping_add(bl).wrapping_add(cr),  // Move this to the end
    ]
}

fn hash_to_bytes(hash: [u32; 5]) -> [u8; 20] {
    let mut output = [0u8; 20];
    seq!(I in 0..5 {
        let bytes = hash[I].to_le_bytes();
        output[I * 4..(I + 1) * 4].copy_from_slice(&bytes);
    });
    output
}

fn input_to_u32(input: &[u8; 32]) -> [u32; 8] {
    let mut u32_input = [0u32; 8];
    seq!(I in 0..8 {
        u32_input[I] = u32::from_le_bytes([
            input[I * 4],
            input[I * 4 + 1],
            input[I * 4 + 2],
            input[I * 4 + 3],
        ]);
    });
    u32_input
}

pub fn ripemd160_32bytes_from_bytes(input: &[u8; 32]) -> [u8; 20] {
    let u32_input = input_to_u32(input);
    let u32_output = ripemd160_32bytes_u32(u32_input);
    hash_to_bytes(u32_output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ripemd160() {
        let input: [u8; 32] = "brandonros/000000000000000000000".as_bytes().try_into().unwrap();
        let result = ripemd160_32bytes_from_bytes(&input);
        let expected: [u8; 20] = hex::decode("cef732cee67ea5d81d08708b22bf1fc7911d3209").unwrap().try_into().unwrap();
        assert_eq!(result, expected);
    }
}