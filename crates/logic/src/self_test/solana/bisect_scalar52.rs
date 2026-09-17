//! Verbatim copy of the parts of `curve25519_dalek::backend::serial::u64::scalar`
//! we need for slot 71's ladder bisect. If the GPU produces wrong
//! results from THIS code (which is identical to what dalek runs
//! internally), the bug is in the compiler's lowering of these
//! specific Rust idioms, not in dalek's API surface.
//!
//! Source: curve25519-dalek 4.1.3.

/// u64 * u64 = u128 multiply helper (dalek's `m`).
#[inline(always)]
fn m(x: u64, y: u64) -> u128 {
    (x as u128) * (y as u128)
}

#[derive(Copy, Clone)]
pub struct Scalar52(pub [u64; 5]);

/// `L` = group order of curve25519's scalar field.
pub const L: Scalar52 = Scalar52([
    0x0002631a5cf5d3ed,
    0x000dea2f79cd6581,
    0x000000000014def9,
    0x0000000000000000,
    0x0000100000000000,
]);

/// `L` * LFACTOR ≡ -1 (mod 2^52)
pub const LFACTOR: u64 = 0x51da312547e1b;

/// `R` = 2^260 mod L
pub const R: Scalar52 = Scalar52([
    0x000f48bd6721e6ed,
    0x0003bab5ac67e45a,
    0x000fffffeb35e51b,
    0x000fffffffffffff,
    0x00000fffffffffff,
]);

impl Scalar52 {
    pub const ZERO: Scalar52 = Scalar52([0, 0, 0, 0, 0]);

    /// Unpack 32 bytes (little-endian) into 5 52-bit limbs.
    pub fn from_bytes(bytes: &[u8; 32]) -> Scalar52 {
        let mut words = [0u64; 4];
        for i in 0..4 {
            for j in 0..8 {
                words[i] |= (bytes[(i * 8) + j] as u64) << (j * 8);
            }
        }
        let mask = (1u64 << 52) - 1;
        let top_mask = (1u64 << 48) - 1;
        let mut s = Scalar52::ZERO;
        s.0[0] = words[0] & mask;
        s.0[1] = ((words[0] >> 52) | (words[1] << 12)) & mask;
        s.0[2] = ((words[1] >> 40) | (words[2] << 24)) & mask;
        s.0[3] = ((words[2] >> 28) | (words[3] << 36)) & mask;
        s.0[4] = (words[3] >> 16) & top_mask;
        s
    }

    /// Pack 5 52-bit limbs into 32 bytes (little-endian).
    #[allow(clippy::identity_op)]
    pub fn as_bytes(&self) -> [u8; 32] {
        let mut s = [0u8; 32];
        s[0] = (self.0[0] >> 0) as u8;
        s[1] = (self.0[0] >> 8) as u8;
        s[2] = (self.0[0] >> 16) as u8;
        s[3] = (self.0[0] >> 24) as u8;
        s[4] = (self.0[0] >> 32) as u8;
        s[5] = (self.0[0] >> 40) as u8;
        s[6] = ((self.0[0] >> 48) | (self.0[1] << 4)) as u8;
        s[7] = (self.0[1] >> 4) as u8;
        s[8] = (self.0[1] >> 12) as u8;
        s[9] = (self.0[1] >> 20) as u8;
        s[10] = (self.0[1] >> 28) as u8;
        s[11] = (self.0[1] >> 36) as u8;
        s[12] = (self.0[1] >> 44) as u8;
        s[13] = (self.0[2] >> 0) as u8;
        s[14] = (self.0[2] >> 8) as u8;
        s[15] = (self.0[2] >> 16) as u8;
        s[16] = (self.0[2] >> 24) as u8;
        s[17] = (self.0[2] >> 32) as u8;
        s[18] = (self.0[2] >> 40) as u8;
        s[19] = ((self.0[2] >> 48) | (self.0[3] << 4)) as u8;
        s[20] = (self.0[3] >> 4) as u8;
        s[21] = (self.0[3] >> 12) as u8;
        s[22] = (self.0[3] >> 20) as u8;
        s[23] = (self.0[3] >> 28) as u8;
        s[24] = (self.0[3] >> 36) as u8;
        s[25] = (self.0[3] >> 44) as u8;
        s[26] = (self.0[4] >> 0) as u8;
        s[27] = (self.0[4] >> 8) as u8;
        s[28] = (self.0[4] >> 16) as u8;
        s[29] = (self.0[4] >> 24) as u8;
        s[30] = (self.0[4] >> 32) as u8;
        s[31] = (self.0[4] >> 40) as u8;
        s
    }

    /// 5×5 widening multiply → [u128; 9]. The exact shape dalek uses.
    pub fn mul_internal(a: &Scalar52, b: &Scalar52) -> [u128; 9] {
        let mut z = [0u128; 9];
        z[0] = m(a.0[0], b.0[0]);
        z[1] = m(a.0[0], b.0[1]) + m(a.0[1], b.0[0]);
        z[2] = m(a.0[0], b.0[2]) + m(a.0[1], b.0[1]) + m(a.0[2], b.0[0]);
        z[3] = m(a.0[0], b.0[3]) + m(a.0[1], b.0[2]) + m(a.0[2], b.0[1]) + m(a.0[3], b.0[0]);
        z[4] = m(a.0[0], b.0[4])
            + m(a.0[1], b.0[3])
            + m(a.0[2], b.0[2])
            + m(a.0[3], b.0[1])
            + m(a.0[4], b.0[0]);
        z[5] = m(a.0[1], b.0[4]) + m(a.0[2], b.0[3]) + m(a.0[3], b.0[2]) + m(a.0[4], b.0[1]);
        z[6] = m(a.0[2], b.0[4]) + m(a.0[3], b.0[3]) + m(a.0[4], b.0[2]);
        z[7] = m(a.0[3], b.0[4]) + m(a.0[4], b.0[3]);
        z[8] = m(a.0[4], b.0[4]);
        z
    }

    /// Compute `a - b` (mod L). Verbatim from dalek's u64 scalar.rs.
    ///
    /// The trailing conditional-add (when the borrow underflows the
    /// low 52 bits, add L back) makes this a constant-time signed-vs-
    /// unsigned bridge. Uses a per-function `black_box` via volatile
    /// load to prevent LLVM from inserting `jns` branches.
    pub fn sub(a: &Scalar52, b: &Scalar52) -> Scalar52 {
        fn black_box(value: u64) -> u64 {
            // Same as dalek: ptr::read_volatile to defeat optimization
            unsafe { core::ptr::read_volatile(&value) }
        }
        let mut difference = Scalar52::ZERO;
        let mask = (1u64 << 52) - 1;
        let mut borrow: u64 = 0;
        for i in 0..5 {
            borrow = a.0[i].wrapping_sub(b.0[i] + (borrow >> 63));
            difference.0[i] = borrow & mask;
        }
        let underflow_mask = ((borrow >> 63) ^ 1).wrapping_sub(1);
        let mut carry: u64 = 0;
        for i in 0..5 {
            carry = (carry >> 52) + difference.0[i] + (L.0[i] & black_box(underflow_mask));
            difference.0[i] = carry & mask;
        }
        difference
    }

    /// Same as `montgomery_reduce` but stops before the final
    /// `Scalar52::sub(result, L)` call. Used by slot 85 — preserves
    /// the test point from the v1.46 run where 85 was confirmed to
    /// PASS without sub. Keeping this separate lets us directly
    /// compare "reduce without sub" (slot 85) vs "reduce with sub"
    /// (slot 90) results on each subsequent vast run.
    pub fn montgomery_reduce_no_sub(limbs: &[u128; 9]) -> Scalar52 {
        #[inline(always)]
        fn part1(sum: u128) -> (u128, u64) {
            let p = (sum as u64).wrapping_mul(LFACTOR) & ((1u64 << 52) - 1);
            ((sum + m(p, L.0[0])) >> 52, p)
        }
        #[inline(always)]
        fn part2(sum: u128) -> (u128, u64) {
            let w = (sum as u64) & ((1u64 << 52) - 1);
            (sum >> 52, w)
        }
        let l = &L;
        let (carry, n0) = part1(limbs[0]);
        let (carry, n1) = part1(carry + limbs[1] + m(n0, l.0[1]));
        let (carry, n2) = part1(carry + limbs[2] + m(n0, l.0[2]) + m(n1, l.0[1]));
        let (carry, n3) = part1(carry + limbs[3] + m(n1, l.0[2]) + m(n2, l.0[1]));
        let (carry, n4) = part1(carry + limbs[4] + m(n0, l.0[4]) + m(n2, l.0[2]) + m(n3, l.0[1]));
        let (carry, r0) = part2(carry + limbs[5] + m(n1, l.0[4]) + m(n3, l.0[2]) + m(n4, l.0[1]));
        let (carry, r1) = part2(carry + limbs[6] + m(n2, l.0[4]) + m(n4, l.0[2]));
        let (carry, r2) = part2(carry + limbs[7] + m(n3, l.0[4]));
        let (carry, r3) = part2(carry + limbs[8] + m(n4, l.0[4]));
        let r4 = carry as u64;
        Scalar52([r0, r1, r2, r3, r4])
    }

    /// Compute `limbs / R` (mod L) — exactly dalek's montgomery_reduce
    /// including the final `Scalar52::sub(result, L)` call.
    pub fn montgomery_reduce(limbs: &[u128; 9]) -> Scalar52 {
        #[inline(always)]
        fn part1(sum: u128) -> (u128, u64) {
            let p = (sum as u64).wrapping_mul(LFACTOR) & ((1u64 << 52) - 1);
            ((sum + m(p, L.0[0])) >> 52, p)
        }
        #[inline(always)]
        fn part2(sum: u128) -> (u128, u64) {
            let w = (sum as u64) & ((1u64 << 52) - 1);
            (sum >> 52, w)
        }
        let l = &L;
        let (carry, n0) = part1(limbs[0]);
        let (carry, n1) = part1(carry + limbs[1] + m(n0, l.0[1]));
        let (carry, n2) = part1(carry + limbs[2] + m(n0, l.0[2]) + m(n1, l.0[1]));
        let (carry, n3) = part1(carry + limbs[3] + m(n1, l.0[2]) + m(n2, l.0[1]));
        let (carry, n4) = part1(carry + limbs[4] + m(n0, l.0[4]) + m(n2, l.0[2]) + m(n3, l.0[1]));
        let (carry, r0) = part2(carry + limbs[5] + m(n1, l.0[4]) + m(n3, l.0[2]) + m(n4, l.0[1]));
        let (carry, r1) = part2(carry + limbs[6] + m(n2, l.0[4]) + m(n4, l.0[2]));
        let (carry, r2) = part2(carry + limbs[7] + m(n3, l.0[4]));
        let (carry, r3) = part2(carry + limbs[8] + m(n4, l.0[4]));
        let r4 = carry as u64;
        // The full dalek implementation: result may be >= L, so
        // attempt to subtract L. This was missing from earlier
        // iterations of the port — slot 86 PASSed without it, which
        // told us mul_internal + reduce-without-sub are fine but
        // hid the fact that sub itself might be the bug.
        Scalar52::sub(&Scalar52([r0, r1, r2, r3, r4]), l)
    }
}
