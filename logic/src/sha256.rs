use seq_macro::seq;

// SHA-256 constants (first 32 bits of the fractional parts of the cube roots of the first 64 primes)
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// SHA-256 initial hash values (first 32 bits of the fractional parts of the square roots of the first 8 primes)
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

#[inline(always)]
const fn rotr32(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}

#[inline(always)]
const fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (!x & z)
}

#[inline(always)]
const fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

#[inline(always)]
const fn big_sigma0(x: u32) -> u32 {
    rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)
}

#[inline(always)]
const fn big_sigma1(x: u32) -> u32 {
    rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)
}

#[inline(always)]
const fn small_sigma0(x: u32) -> u32 {
    rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3)
}

#[inline(always)]
const fn small_sigma1(x: u32) -> u32 {
    rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10)
}

fn sha256_32(input: [u32; 8]) -> [u32; 8] {
    // Message schedule array - we need 64 words for SHA-256
    let mut w = [0u32; 64];
    
    // Copy input to first 8 words (big-endian)
    seq!(I in 0..8 {
        w[I] = input[I].to_be();
    });
    
    // Add padding: single 1 bit followed by zeros, then length
    // For 32 bytes (256 bits), padding is: 0x80000000, then zeros, then length = 256
    w[8] = 0x80000000;
    // w[9] through w[13] are already zero
    w[14] = 0; // Upper 32 bits of length (always 0 for our use case)
    w[15] = 256; // Lower 32 bits of length (32 bytes = 256 bits)
    
    // Extend the first 16 words into the remaining 48 words
    seq!(N in 16..64 {
        w[N] = small_sigma1(w[N - 2])
            .wrapping_add(w[N - 7])
            .wrapping_add(small_sigma0(w[N - 15]))
            .wrapping_add(w[N - 16]);
    });
    
    // Initialize working variables
    let mut a = H0[0];
    let mut b = H0[1];
    let mut c = H0[2];
    let mut d = H0[3];
    let mut e = H0[4];
    let mut f = H0[5];
    let mut g = H0[6];
    let mut h = H0[7];
    
    // Main loop - 64 rounds
    seq!(ROUND in 0..64 {
        let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(K[ROUND])
            .wrapping_add(w[ROUND]);
        
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
        
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    });
    
    // Add this chunk's hash to the result so far
    [
        H0[0].wrapping_add(a),
        H0[1].wrapping_add(b),
        H0[2].wrapping_add(c),
        H0[3].wrapping_add(d),
        H0[4].wrapping_add(e),
        H0[5].wrapping_add(f),
        H0[6].wrapping_add(g),
        H0[7].wrapping_add(h),
    ]
}

fn sha256_variable_length(input: &[u8]) -> [u32; 8] {
    // Message schedule array - we need 64 words for SHA-256
    let mut w = [0u32; 64];
    
    let input_len = input.len();
    let input_bits = input_len * 8;
    
    // Convert bytes to u32 words (big-endian)
    let full_words = input_len / 4;
    for i in 0..full_words {
        w[i] = u32::from_be_bytes([
            input[i * 4],
            input[i * 4 + 1], 
            input[i * 4 + 2],
            input[i * 4 + 3]
        ]);
    }
    
    // Handle remaining bytes (if input length is not multiple of 4)
    let remaining_bytes = input_len % 4;
    if remaining_bytes > 0 {
        let mut last_word = [0u8; 4];
        for i in 0..remaining_bytes {
            last_word[i] = input[full_words * 4 + i];
        }
        // Add the padding bit (0x80) right after the last byte
        last_word[remaining_bytes] = 0x80;
        w[full_words] = u32::from_be_bytes(last_word);
    } else {
        // If input is exactly multiple of 4, add padding in next word
        w[full_words] = 0x80000000;
    }
    
    // Add length at the end (last 64 bits = words 14 and 15)
    w[14] = 0; // Upper 32 bits of length (always 0 for inputs < 2^32 bits)
    w[15] = input_bits as u32; // Lower 32 bits of length
    
    // Extend the first 16 words into the remaining 48 words
    seq!(N in 16..64 {
        w[N] = small_sigma1(w[N - 2])
            .wrapping_add(w[N - 7])
            .wrapping_add(small_sigma0(w[N - 15]))
            .wrapping_add(w[N - 16]);
    });
    
    // Initialize working variables
    let mut a = H0[0];
    let mut b = H0[1];
    let mut c = H0[2];
    let mut d = H0[3];
    let mut e = H0[4];
    let mut f = H0[5];
    let mut g = H0[6];
    let mut h = H0[7];
    
    // Main loop - 64 rounds
    seq!(ROUND in 0..64 {
        let t1 = h
            .wrapping_add(big_sigma1(e))
            .wrapping_add(ch(e, f, g))
            .wrapping_add(K[ROUND])
            .wrapping_add(w[ROUND]);
        
        let t2 = big_sigma0(a).wrapping_add(maj(a, b, c));
        
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    });
    
    // Add this chunk's hash to the result so far
    [
        H0[0].wrapping_add(a),
        H0[1].wrapping_add(b),
        H0[2].wrapping_add(c),
        H0[3].wrapping_add(d),
        H0[4].wrapping_add(e),
        H0[5].wrapping_add(f),
        H0[6].wrapping_add(g),
        H0[7].wrapping_add(h),
    ]
}

fn hash_to_bytes(hash: [u32; 8]) -> [u8; 32] {
    let mut output = [0u8; 32];
    seq!(I in 0..8 {
        let bytes = hash[I].to_be_bytes();
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

pub fn sha256_32_from_bytes(input: &[u8; 32]) -> [u8; 32] {
    let u32_input = input_to_u32(input);
    let u32_output = sha256_32(u32_input);
    hash_to_bytes(u32_output)
}

pub fn sha256_from_bytes(input: &[u8]) -> [u8; 32] {
    let hash_words = sha256_variable_length(input);
    let mut result = [0u8; 32];
    seq!(N in 0..8 {
        let bytes = hash_words[N].to_be_bytes();
        result[N * 4..N * 4 + 4].copy_from_slice(&bytes);
    });
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256() {
        // Test with 32 zero bytes
        let input: [u8; 32] = "brandonros/000000000000000000000".as_bytes().try_into().unwrap();
        let result = sha256_32_from_bytes(&input);
        let expected: [u8; 32] = hex::decode("f7a41dae1196282f0a544a8c7f1bbf61bda79307dc424c0d9febd27b08e1bf78").unwrap().try_into().unwrap();
        assert_eq!(result, expected);
    }
}
