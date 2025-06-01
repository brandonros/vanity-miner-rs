use seq_macro::seq;

const K: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

const H0: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

#[inline(always)]
const fn rotr64(x: u64, n: u32) -> u64 {
    (x >> n) | (x << (64 - n))
}

#[inline(always)]
const fn ch(x: u64, y: u64, z: u64) -> u64 {
    (x & y) ^ (!x & z)
}

#[inline(always)]
const fn maj(x: u64, y: u64, z: u64) -> u64 {
    (x & y) ^ (x & z) ^ (y & z)
}

#[inline(always)]
const fn big_sigma0(x: u64) -> u64 {
    rotr64(x, 28) ^ rotr64(x, 34) ^ rotr64(x, 39)
}

#[inline(always)]
const fn big_sigma1(x: u64) -> u64 {
    rotr64(x, 14) ^ rotr64(x, 18) ^ rotr64(x, 41)
}

#[inline(always)]
const fn small_sigma0(x: u64) -> u64 {
    rotr64(x, 1) ^ rotr64(x, 8) ^ (x >> 7)
}

#[inline(always)]
const fn small_sigma1(x: u64) -> u64 {
    rotr64(x, 19) ^ rotr64(x, 61) ^ (x >> 6)
}

fn sha512_32bytes_u64(input: [u64; 4]) -> [u64; 8] {
    // Message schedule array - we only need 16 words since our message is exactly 32 bytes
    let mut w = [0u64; 80];
    
    // Copy input to first 4 words (big-endian)
    w[0] = input[0].to_be();
    w[1] = input[1].to_be();
    w[2] = input[2].to_be();
    w[3] = input[3].to_be();
    
    // Add padding: single 1 bit followed by zeros, then length
    // For 32 bytes (256 bits), padding is: 0x8000000000000000, then zeros, then length = 256
    w[4] = 0x8000000000000000;
    // w[5] through w[13] are already zero
    w[14] = 0; // Upper 64 bits of length (always 0 for our use case)
    w[15] = 256; // Lower 64 bits of length (32 bytes = 256 bits)
    
    // Extend the first 16 words into the remaining 64 words
    seq!(N in 16..80 {
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
    
    // Main loop - 80 rounds
    seq!(ROUND in 0..80 {
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

/// Convert hash output back to bytes
fn hash_to_bytes(hash: [u64; 8]) -> [u8; 64] {
    let mut output = [0u8; 64];
    seq!(I in 0..8 {
        let bytes = hash[I].to_be_bytes();
        output[I * 8..(I + 1) * 8].copy_from_slice(&bytes);
    });
    output
}

fn input_to_u64(input: &[u8; 32]) -> [u64; 4] {
    let mut u64_input = [0u64; 4];
    seq!(I in 0..4 {
        u64_input[I] = u64::from_le_bytes([
            input[I * 8],
            input[I * 8 + 1],
            input[I * 8 + 2],
            input[I * 8 + 3],
            input[I * 8 + 4],
            input[I * 8 + 5],
            input[I * 8 + 6],
            input[I * 8 + 7],
        ]);
    });
    u64_input
}

pub fn sha512_32bytes_from_bytes(input: &[u8; 32]) -> [u8; 64] {
    let u64_input = input_to_u64(input);
    let u64_output = sha512_32bytes_u64(u64_input);
    hash_to_bytes(u64_output)
}
