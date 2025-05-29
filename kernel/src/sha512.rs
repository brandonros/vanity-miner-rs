//! A small, self-contained SHA512 implementation
//! (C) Frank Denis, public domain

#![allow(
    non_snake_case,
    clippy::cast_lossless,
    clippy::eq_op,
    clippy::identity_op,
    clippy::many_single_char_names,
    clippy::unreadable_literal
)]

const ROUND_CONSTANTS: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

const INITIAL_STATE: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179
];

#[inline(always)]
fn load_be(base: &[u8], offset: usize) -> u64 {
    let addr = &base[offset..];
    (addr[7] as u64)
        | (addr[6] as u64) << 8
        | (addr[5] as u64) << 16
        | (addr[4] as u64) << 24
        | (addr[3] as u64) << 32
        | (addr[2] as u64) << 40
        | (addr[1] as u64) << 48
        | (addr[0] as u64) << 56
}

#[inline(always)]
fn store_be(base: &mut [u8], offset: usize, x: u64) {
    let addr = &mut base[offset..];
    addr[7] = x as u8;
    addr[6] = (x >> 8) as u8;
    addr[5] = (x >> 16) as u8;
    addr[4] = (x >> 24) as u8;
    addr[3] = (x >> 32) as u8;
    addr[2] = (x >> 40) as u8;
    addr[1] = (x >> 48) as u8;
    addr[0] = (x >> 56) as u8;
}

struct W([u64; 16]);

struct State([u64; 8]);

impl W {
    #[inline(always)]
    fn new(input: &[u8]) -> Self {
        let mut w = [0u64; 16];
        for (i, e) in w.iter_mut().enumerate() {
            *e = load_be(input, i * 8)
        }
        W(w)
    }

    #[inline(always)]
    fn Ch(x: u64, y: u64, z: u64) -> u64 {
        (x & y) ^ (!x & z)
    }

    #[inline(always)]
    fn Maj(x: u64, y: u64, z: u64) -> u64 {
        (x & y) ^ (x & z) ^ (y & z)
    }

    #[inline(always)]
    fn Sigma0(x: u64) -> u64 {
        x.rotate_right(28) ^ x.rotate_right(34) ^ x.rotate_right(39)
    }

    #[inline(always)]
    fn Sigma1(x: u64) -> u64 {
        x.rotate_right(14) ^ x.rotate_right(18) ^ x.rotate_right(41)
    }

    #[inline(always)]
    fn sigma0(x: u64) -> u64 {
        x.rotate_right(1) ^ x.rotate_right(8) ^ (x >> 7)
    }

    #[inline(always)]
    fn sigma1(x: u64) -> u64 {
        x.rotate_right(19) ^ x.rotate_right(61) ^ (x >> 6)
    }

    #[inline(always)]
    fn M(&mut self, a: usize, b: usize, c: usize, d: usize) {
        let w = &mut self.0;
        w[a] = w[a]
            .wrapping_add(Self::sigma1(w[b]))
            .wrapping_add(w[c])
            .wrapping_add(Self::sigma0(w[d]));
    }

    #[inline(always)]
    fn expand(&mut self) {
        for i in 0..16 {
            self.M(i, (i + 14) & 15, (i + 9) & 15, (i + 1) & 15);
        }
    }

    #[inline(always)]
    fn F(&mut self, state: &mut State, i: usize, k: u64) {
        let t = &mut state.0;
        
        // Pre-calculate all indices to improve readability
        let idx_7 = (16 - i + 7) & 7;
        let idx_6 = (16 - i + 6) & 7;
        let idx_5 = (16 - i + 5) & 7;
        let idx_4 = (16 - i + 4) & 7;
        let idx_3 = (16 - i + 3) & 7;
        let idx_2 = (16 - i + 2) & 7;
        let idx_1 = (16 - i + 1) & 7;
        let idx_0 = (16 - i + 0) & 7;
        
        // Update using pre-calculated indices
        t[idx_7] = t[idx_7]
            .wrapping_add(Self::Sigma1(t[idx_4]))
            .wrapping_add(Self::Ch(t[idx_4], t[idx_5], t[idx_6]))
            .wrapping_add(k)
            .wrapping_add(self.0[i]);
        t[idx_3] = t[idx_3].wrapping_add(t[idx_7]);
        t[idx_7] = t[idx_7]
            .wrapping_add(Self::Sigma0(t[idx_0]))
            .wrapping_add(Self::Maj(t[idx_0], t[idx_1], t[idx_2]));
    }

    #[inline(always)]
    fn G(&mut self, state: &mut State, s: usize) {
        let rc = &ROUND_CONSTANTS[s * 16..];
        for i in 0..16 {
            self.F(state, i, rc[i]);
        }
    }
}

impl State {
    fn new() -> Self {
        State(INITIAL_STATE)
    }

    #[inline(always)]
    fn add(&mut self, x: &State) {
        let sx = &mut self.0;
        let ex = &x.0;
        for i in 0..8 {
            sx[i] = sx[i].wrapping_add(ex[i]);
        }
    }

    #[inline(always)]
    fn store(&self, out: &mut [u8]) {
        for (i, &e) in self.0.iter().enumerate() {
            store_be(out, i * 8, e);
        }
    }

    #[inline(always)]
    fn blocks(&mut self, mut input: &[u8]) -> usize {
        let mut t = State([
            self.0[0], 
            self.0[1], 
            self.0[2], 
            self.0[3], 
            self.0[4], 
            self.0[5], 
            self.0[6], 
            self.0[7]
        ]);
        let mut inlen = input.len();
        while inlen >= 128 {
            let mut w = W::new(input);
            for i in 0..5 {
                w.G(&mut t, i);
                w.expand();
            }
            t.add(self);
            self.0 = t.0;
            input = &input[128..];
            inlen -= 128;
        }
        inlen
    }
}

pub struct Hasher {
    state: State,
    w: [u8; 128],
    r: usize,
    len: usize,
}

impl Hasher {
    pub fn new() -> Hasher {
        Hasher {
            state: State::new(),
            r: 0,
            w: [0u8; 128],
            len: 0,
        }
    }

    /// Absorb content
    pub fn update<T: AsRef<[u8]>>(&mut self, input: T) {
        let input = input.as_ref();
        let mut n = input.len();
        self.len += n;
        let av = 128 - self.r;
        let tc = ::core::cmp::min(n, av);
        self.w[self.r..self.r + tc].copy_from_slice(&input[0..tc]);
        self.r += tc;
        n -= tc;
        let pos = tc;
        if self.r == 128 {
            self.state.blocks(&self.w);
            self.r = 0;
        }
        if self.r == 0 && n > 0 {
            let rb = self.state.blocks(&input[pos..]);
            if rb > 0 {
                self.w[..rb].copy_from_slice(&input[pos + n - rb..]);
                self.r = rb;
            }
        }
    }

    /// Compute SHA512(absorbed content)
    pub fn finalize(mut self) -> [u8; 64] {
        let mut padded = [0u8; 256];
        padded[..self.r].copy_from_slice(&self.w[..self.r]);
        padded[self.r] = 0x80;
        let r = if self.r < 112 { 128 } else { 256 };
        let bits = self.len * 8;
        for i in 0..8 {
            padded[r - 8 + i] = (bits as u64 >> (56 - i * 8)) as u8;
        }
        self.state.blocks(&padded[..r]);
        let mut out = [0u8; 64];
        self.state.store(&mut out);
        out
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}
