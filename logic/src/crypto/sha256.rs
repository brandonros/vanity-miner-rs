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
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
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

fn process_block(block: &[u32; 16], state: &mut [u32; 8]) {
    // Message schedule array - we need 64 words for SHA-256
    let mut w = [0u32; 64];

    // Copy input block
    w[0..16].copy_from_slice(block);

    // Extend the first 16 words into the remaining 48 words
    seq!(N in 16..64 {
        w[N] = small_sigma1(w[N - 2])
            .wrapping_add(w[N - 7])
            .wrapping_add(small_sigma0(w[N - 15]))
            .wrapping_add(w[N - 16]);
    });

    // Initialize working variables
    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

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
    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
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
    let input_len = input.len();
    let input_bits = (input_len as u64) * 8;

    // Initialize hash state
    let mut state = H0;

    // Process complete 64-byte blocks
    let complete_blocks = input_len / 64;
    for block_idx in 0..complete_blocks {
        let mut block = [0u32; 16];
        let start = block_idx * 64;

        // Convert 64 bytes to 16 u32 words (big-endian)
        for i in 0..16 {
            let byte_idx = start + i * 4;
            block[i] = u32::from_be_bytes([
                input[byte_idx],
                input[byte_idx + 1],
                input[byte_idx + 2],
                input[byte_idx + 3],
            ]);
        }

        process_block(&block, &mut state);
    }

    // Handle the final partial block with padding
    let remaining_bytes = input_len % 64;
    let remaining_start = complete_blocks * 64;

    let mut final_block = [0u32; 16];

    // Convert remaining bytes to words
    let complete_words_in_final = remaining_bytes / 4;
    for i in 0..complete_words_in_final {
        let byte_idx = remaining_start + i * 4;
        final_block[i] = u32::from_be_bytes([
            input[byte_idx],
            input[byte_idx + 1],
            input[byte_idx + 2],
            input[byte_idx + 3],
        ]);
    }

    // Handle the last partial word (if any) and add padding
    let remaining_bytes_in_word = remaining_bytes % 4;
    if remaining_bytes_in_word > 0 {
        let mut last_word_bytes = [0u8; 4];
        for i in 0..remaining_bytes_in_word {
            last_word_bytes[i] = input[remaining_start + complete_words_in_final * 4 + i];
        }
        // Add the padding bit (0x80) right after the last byte
        last_word_bytes[remaining_bytes_in_word] = 0x80;
        final_block[complete_words_in_final] = u32::from_be_bytes(last_word_bytes);
    } else {
        // remaining_bytes ∈ [0,63] ⇒ complete_words_in_final = remaining_bytes/4 ∈ [0,15],
        // so this index is always in range and the "no room" branch is unreachable.
        final_block[complete_words_in_final] = 0x80000000;
    }

    // If 0x80 lands in word 14 or 15, the 8-byte length (words 14 and 15)
    // doesn't fit in this block — push it and start a fresh padding block.
    if complete_words_in_final + 1 > 14 {
        process_block(&final_block, &mut state);
        final_block = [0u32; 16];
    }

    // Add length in bits as the last 64 bits (big-endian)
    final_block[14] = (input_bits >> 32) as u32; // Upper 32 bits
    final_block[15] = input_bits as u32; // Lower 32 bits

    process_block(&final_block, &mut state);

    state
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

/// Incremental SHA-256 using the same compression function as the one-shot paths.
/// Updates borrow their input; only the last partial block is buffered.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buffer: [u8; 64],
    used: usize,
    bytes: u64,
}

impl Sha256 {
    pub const fn new() -> Self {
        Self {
            state: H0,
            buffer: [0; 64],
            used: 0,
            bytes: 0,
        }
    }

    pub fn digest(input: impl AsRef<[u8]>) -> [u8; 32] {
        sha256_from_bytes(input.as_ref())
    }

    pub fn update(&mut self, input: impl AsRef<[u8]>) {
        self.update_slice(input.as_ref());
    }

    fn update_slice(&mut self, mut input: &[u8]) {
        self.bytes = self.bytes.wrapping_add(input.len() as u64);
        if self.used != 0 {
            let take = (64 - self.used).min(input.len());
            self.buffer[self.used..self.used + take].copy_from_slice(&input[..take]);
            self.used += take;
            input = &input[take..];
            if self.used != 64 {
                return;
            }
            Self::compress(&mut self.state, &self.buffer);
            self.used = 0;
        }
        while input.len() >= 64 {
            Self::compress(&mut self.state, &input[..64]);
            input = &input[64..];
        }
        self.buffer[..input.len()].copy_from_slice(input);
        self.used = input.len();
    }

    fn compress(state: &mut [u32; 8], bytes: &[u8]) {
        let mut block = [0u32; 16];
        for (word, chunk) in block.iter_mut().zip(bytes.chunks_exact(4)) {
            *word = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        process_block(&block, state);
    }

    pub fn finalize(mut self) -> [u8; 32] {
        self.buffer[self.used] = 0x80;
        self.buffer[self.used + 1..].fill(0);
        if self.used >= 56 {
            Self::compress(&mut self.state, &self.buffer);
            self.buffer = [0; 64];
        }
        self.buffer[56..].copy_from_slice(&self.bytes.wrapping_mul(8).to_be_bytes());
        Self::compress(&mut self.state, &self.buffer);
        hash_to_bytes(self.state)
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

// The crypto-search callers use RustCrypto's HMAC and RSA digest interfaces.
// Bitcoin/Shallenge's existing paths do not enable this dependency.
#[cfg(feature = "crypto-search")]
mod digest_traits {
    use super::Sha256;
    use digest::{
        FixedOutput, FixedOutputReset, HashMarker, Output, OutputSizeUser, Reset, Update,
    };
    impl HashMarker for Sha256 {}
    impl OutputSizeUser for Sha256 {
        type OutputSize = digest::consts::U32;
    }
    impl digest::core_api::BlockSizeUser for Sha256 {
        type BlockSize = digest::consts::U64;
    }
    impl Update for Sha256 {
        fn update(&mut self, data: &[u8]) {
            self.update_slice(data);
        }
    }
    impl FixedOutput for Sha256 {
        fn finalize_into(self, out: &mut Output<Self>) {
            out.copy_from_slice(&self.finalize());
        }
    }
    impl Reset for Sha256 {
        fn reset(&mut self) {
            *self = Self::new();
        }
    }
    impl FixedOutputReset for Sha256 {
        fn finalize_into_reset(&mut self, out: &mut Output<Self>) {
            out.copy_from_slice(&core::mem::take(self).finalize());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_32() {
        // Test with 32 bytes
        let input: [u8; 32] = "brandonros/000000000000000000000"
            .as_bytes()
            .try_into()
            .unwrap();
        let result = sha256_32_from_bytes(&input);
        let expected: [u8; 32] =
            hex::decode("f7a41dae1196282f0a544a8c7f1bbf61bda79307dc424c0d9febd27b08e1bf78")
                .unwrap()
                .try_into()
                .unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sha256_variable() {
        // Test with 33 bytes
        let input: [u8; 33] = "brandonros/0000000000000000000000"
            .as_bytes()
            .try_into()
            .unwrap();
        let result = sha256_from_bytes(&input);
        let expected: [u8; 32] =
            hex::decode("062389936c519ed73f3371ef2e66d438e1cf0a6603f8b67c748a5d211e48b29d")
                .unwrap()
                .try_into()
                .unwrap();
        assert_eq!(result, expected);
    }

    // 112-byte test vector: one full block and a final padded block.
    // Exercises the full-block loop and final-block padding.
    #[test]
    fn test_sha256_multi_block() {
        let input = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
        assert_eq!(input.len(), 112);
        let result = sha256_from_bytes(input);
        let expected: [u8; 32] =
            hex::decode("cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1")
                .unwrap()
                .try_into()
                .unwrap();
        assert_eq!(result, expected);
    }

    // NIST FIPS 180-2 vector at the padding boundary: 56-byte input means
    // len % 64 == 56, so the 1-byte 0x80 marker + 8-byte length can't fit
    // in the same block and a second padding block is required.
    #[test]
    fn test_sha256_padding_overflow() {
        let input = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(input.len(), 56);
        let result = sha256_from_bytes(input);
        let expected: [u8; 32] =
            hex::decode("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
                .unwrap()
                .try_into()
                .unwrap();
        assert_eq!(result, expected);
    }

    // Cross-validation against Python's hashlib for long inputs that
    // hammer the multi-block loop. Expected values computed via:
    //   python3 -c "import hashlib; print(hashlib.sha256(MSG).hexdigest())"
    #[test]
    fn test_sha256_256_bytes_of_a() {
        let input = [b'a'; 256];
        let result = sha256_from_bytes(&input);
        let expected: [u8; 32] =
            hex::decode("02d7160d77e18c6447be80c2e355c7ed4388545271702c50253b0914c65ce5fe")
                .unwrap()
                .try_into()
                .unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sha256_256_distinct_bytes() {
        let mut input = [0u8; 256];
        for i in 0..256 {
            input[i] = i as u8;
        }
        let result = sha256_from_bytes(&input);
        let expected: [u8; 32] =
            hex::decode("40aff2e9d2d8922e47afd4648e6967497158785fbd1da870e7110266bf944880")
                .unwrap()
                .try_into()
                .unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_sha256_1024_bytes() {
        let input = [b'a'; 1024];
        let result = sha256_from_bytes(&input);
        let expected: [u8; 32] =
            hex::decode("2edc986847e209b4016e141a6dc8716d3207350f416969382d431539bf292e4a")
                .unwrap()
                .try_into()
                .unwrap();
        assert_eq!(result, expected);
    }

    // Fixed vectors generated independently with Python hashlib.sha256(bytes(range(len))).
    #[test]
    fn test_sha256_block_and_padding_boundaries() {
        let cases = [
            (
                0,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                1,
                "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
            ),
            (
                55,
                "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59",
            ),
            (
                56,
                "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562",
            ),
            (
                63,
                "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488",
            ),
            (
                64,
                "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108",
            ),
            (
                65,
                "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781",
            ),
            (
                119,
                "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6",
            ),
            (
                120,
                "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c",
            ),
            (
                127,
                "92ca0fa6651ee2f97b884b7246a562fa71250fedefe5ebf270d31c546bfea976",
            ),
            (
                128,
                "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5",
            ),
        ];
        let mut input = [0u8; 128];
        for (i, byte) in input.iter_mut().enumerate() {
            *byte = i as u8;
        }
        for (len, expected) in cases {
            assert_eq!(
                hex::encode(sha256_from_bytes(&input[..len])),
                expected,
                "length {len}"
            );
        }
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[test]
    fn every_split_around_block_and_padding_boundaries() {
        let input: [u8; 257] = core::array::from_fn(|i| i as u8);
        for len in 0..=input.len() {
            let expected = sha256_from_bytes(&input[..len]);
            for split in 0..=len {
                let mut hash = Sha256::new();
                hash.update([]);
                hash.update(&input[..split]);
                hash.update([]);
                hash.update(&input[split..len]);
                hash.update([]);
                assert_eq!(hash.finalize(), expected, "length {len}, split {split}");
            }
        }
    }

    #[test]
    fn large_message_in_different_chunk_sizes() {
        let input: alloc::vec::Vec<u8> = (0..102400).map(|i| (i * 37) as u8).collect();
        let expected = sha256_from_bytes(&input);
        for chunk_size in [1, 7, 31, 55, 56, 63, 64, 65, 127, 1024, 65536] {
            let mut hash = Sha256::new();
            for chunk in input.chunks(chunk_size) {
                hash.update(chunk);
            }
            assert_eq!(hash.finalize(), expected, "chunk size {chunk_size}");
        }
    }

    #[test]
    fn published_million_a_vector() {
        let mut hash = Sha256::new();
        for _ in 0..1000 {
            hash.update([b'a'; 1000]);
        }
        assert_eq!(
            hex::encode(hash.finalize()),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[cfg(feature = "crypto-search")]
    #[test]
    fn digest_reset_and_cloned_prefixes() {
        let mut hash = Sha256::new();
        hash.update(b"shared prefix:");
        let mut other = hash.clone();
        hash.update(b"a");
        other.update(b"b");
        assert_eq!(hash.finalize(), sha256_from_bytes(b"shared prefix:a"));
        let result = digest::Digest::finalize_reset(&mut other);
        assert_eq!(&result[..], &sha256_from_bytes(b"shared prefix:b"));
        assert_eq!(other.clone().finalize(), sha256_from_bytes(b""));
        other.update(b"discard this");
        digest::Reset::reset(&mut other);
        other.update(b"abc");
        assert_eq!(
            hex::encode(other.finalize()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[cfg(feature = "crypto-search")]
    #[test]
    fn rfc4231_hmac_short_and_long_keys() {
        use hmac::{Mac, SimpleHmac};
        let mut mac = SimpleHmac::<Sha256>::new_from_slice(&[0x0b; 20]).unwrap();
        mac.update(b"Hi There");
        assert_eq!(
            hex::encode(mac.finalize().into_bytes()),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        let mut mac = SimpleHmac::<Sha256>::new_from_slice(&[0xaa; 131]).unwrap();
        mac.update(b"Test Using Larger Than Block-Size Key - Hash Key First");
        assert_eq!(
            hex::encode(mac.finalize().into_bytes()),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }
}
