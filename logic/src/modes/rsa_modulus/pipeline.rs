//! Device-resident RSA searches. A slot owns one independent factor and a
//! nonrepeating q progression. Launch boundaries publish changes between stages.
use crate::search::candidate_derivation::{CandidateDeriver, CandidateDomain};
use crypto_bigint::{Encoding, NonZero, U1024, U2048};
use zeroize::{Zeroize, Zeroizing};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SearchConfig {
    pub lower: [u8; 256],
    pub upper: [u8; 256],
    pub p_min: [u8; 128],
    pub p_count: [u8; 128],
    pub suffix: [u8; 256],
    pub seed: [u8; 32],
    pub worker: u64,
    pub suffix_bits: u32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Task {
    pub p: [u8; 128],
    pub first: [u8; 128],
    pub count: [u8; 128],
    pub cursor: [u8; 128],
    pub remaining: [u8; 128],
    /// Indices in the original progression excluded by factor separation.
    pub skip_start: [u8; 128],
    pub skip_count: [u8; 128],
    pub id: u64,
    /// 0 = empty, 1 = probable p awaiting a range, 2 = active,
    /// 3 = unchecked p awaiting a range before expensive primality testing.
    pub state: u32,
    /// Copied from the separate atomic winner buffer after the q stage completes.
    pub winner: u32,
}
impl Task {
    pub const EMPTY: Self = Self {
        p: [0; 128],
        first: [0; 128],
        count: [0; 128],
        cursor: [0; 128],
        remaining: [0; 128],
        skip_start: [0; 128],
        skip_count: [0; 128],
        id: 0,
        state: 0,
        winner: 0,
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Pair {
    pub p: [u8; 128],
    pub q: [u8; 128],
    pub id: u64,
}
impl Pair {
    pub const EMPTY: Self = Self {
        p: [0; 128],
        q: [0; 128],
        id: 0,
    };
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Counts {
    pub p_tested: u32,
    pub p_accepted: u32,
    pub ranges: u32,
    pub q_tested: u32,
    pub matches: u32,
    pub errors: u32,
    pub active: u32,
    pub reserved: u32,
}

macro_rules! record {
    ($ty:ty; $($field:ident),+) => {
        impl Zeroize for $ty {
            fn zeroize(&mut self) { $(self.$field.zeroize();)+ }
        }
        // SAFETY: repr(C), integer fields only, no implicit padding (tested below).
        unsafe impl crate::search::device_record::DeviceRecord for $ty {}
    }
}
record!(SearchConfig; lower, upper, p_min, p_count, suffix, seed, worker, suffix_bits, reserved);
record!(Task; p, first, count, cursor, remaining, skip_start, skip_count, id, state, winner);
record!(Pair; p, q, id);
record!(Counts; p_tested, p_accepted, ranges, q_tested, matches, errors, active, reserved);

/// Exact rejection sampling. Four disjoint PRF inputs supply the 1024 bits;
/// neither factor generation nor q-range starts reduce random bytes modulo n.
fn sample(config: &SearchConfig, id: u64, domain: CandidateDomain, bound: &U1024) -> Option<U1024> {
    if *bound == U1024::ZERO {
        return None;
    }
    if *bound == U1024::ONE {
        return Some(U1024::ZERO);
    }
    let bits = bound.wrapping_sub(&U1024::ONE).bits_vartime();
    let mask = U1024::MAX.shr_vartime(1024 - bits);
    let deriver = CandidateDeriver::new(config.seed, domain, [0; 32], [0; 32]);
    // At least half the masked space is valid. Exhaustion is an error, never a
    // biased fallback or a repeated task identifier.
    for attempt in 0..128 {
        let mut bytes = Zeroizing::new([0u8; 128]);
        for block in 0..4 {
            let mut value = deriver.block(
                config.worker,
                (u128::from(id) << 2) | block as u128,
                attempt,
            );
            bytes[block * 32..(block + 1) * 32].copy_from_slice(&value);
            value.zeroize();
        }
        let value = U1024::from_be_slice(bytes.as_ref()).bitand(&mask);
        if value < *bound {
            return Some(value);
        }
    }
    None
}

pub fn generate_p(config: &SearchConfig, id: u64) -> Option<[u8; 128]> {
    let min = U1024::from_be_slice(&config.p_min);
    let count = U1024::from_be_slice(&config.p_count);
    let offset = sample(config, id, CandidateDomain::RsaFactor, &count)?;
    let value: U2048 = min
        .resize::<{ U2048::LIMBS }>()
        .wrapping_add(&offset.resize::<{ U2048::LIMBS }>().shl_vartime(1));
    if value.bits_vartime() != 1024 || value.to_be_bytes()[255] & 1 == 0 {
        return None;
    }
    Some(value.resize::<{ U1024::LIMBS }>().to_be_bytes())
}

pub fn probable_p(p: &[u8; 128]) -> bool {
    crate::crypto::rsa_prime::probable_prime(&U1024::from_be_slice(p))
}

/// Near/beyond 128 constrained bytes, most p's have at most one eligible q.
/// Reject their empty ranges before spending modular exponentiations on p.
/// Broad ranges use the prime-first path to avoid constructing ranges for the
/// many p's cheaply rejected by the prime filter. This is a scheduling heuristic.
pub fn range_first(config: &SearchConfig) -> bool {
    let width =
        U2048::from_be_slice(&config.upper).wrapping_sub(&U2048::from_be_slice(&config.lower));
    width.bits_vartime() <= 1024 + config.suffix_bits as usize
}

/// Exact interval/residue construction. Quotients stay wide until bounds have
/// been checked; ceil division cannot overflow at a full-width modulus.
pub fn progression(config: &SearchConfig, p_bytes: &[u8; 128]) -> Option<([u8; 128], [u8; 128])> {
    let bits = config.suffix_bits as usize;
    if bits == 0 || bits > 2048 || p_bytes[0] & 128 == 0 || p_bytes[127] & 1 == 0 {
        return None;
    }
    let p: U2048 = U1024::from_be_slice(p_bytes).resize();
    let divisor = Option::<NonZero<U2048>>::from(NonZero::new(p))?;
    let (quotient, remainder) = U2048::from_be_slice(&config.lower).div_rem(&divisor);
    let ceil = if remainder == U2048::ZERO {
        quotient
    } else {
        quotient.wrapping_add(&U2048::ONE)
    };
    let min = ceil.max(U2048::ONE.shl_vartime(1023));
    let max = U2048::from_be_slice(&config.upper)
        .div_rem(&divisor)
        .0
        .min(U1024::MAX.resize());
    if min > max {
        return None;
    }
    let mask = U2048::MAX.shr_vartime(2048 - bits);
    let inverse = p.bitand(&mask).inv_mod2k_vartime(bits);
    let residue = U2048::from_be_slice(&config.suffix)
        .wrapping_mul(&inverse)
        .bitand(&mask);
    let delta = residue.wrapping_sub(&min).bitand(&mask);
    let first = min.wrapping_add(&delta);
    if first < min || first > max {
        return None;
    }
    let count = if bits >= 1024 {
        U2048::ONE
    } else {
        max.wrapping_sub(&first)
            .shr_vartime(bits)
            .wrapping_add(&U2048::ONE)
    };
    Some((
        first.resize::<{ U1024::LIMBS }>().to_be_bytes(),
        count.resize::<{ U1024::LIMBS }>().to_be_bytes(),
    ))
}

pub fn prepare_range(config: &SearchConfig, task: &mut Task) -> Result<bool, &'static str> {
    let Some((first, count)) = progression(config, &task.p) else {
        task.zeroize();
        return Ok(false);
    };
    let first_wide: U2048 = U1024::from_be_slice(&first).resize();
    let total: U2048 = U1024::from_be_slice(&count).resize();
    let bits = config.suffix_bits as usize;
    let step = if bits >= 1024 {
        U2048::ZERO
    } else {
        total.wrapping_sub(&U2048::ONE).shl_vartime(bits)
    };
    let last = first_wide.wrapping_add(&step);
    let p: U2048 = U1024::from_be_slice(&task.p).resize();
    let distance = U2048::ONE.shl_vartime(924);
    let forbidden_low = p.wrapping_sub(&distance).max(first_wide);
    let forbidden_high = p.wrapping_add(&distance).min(last);
    let mut skip_start = U2048::ZERO;
    let mut skip_count = U2048::ZERO;
    if forbidden_low <= forbidden_high {
        if bits >= 1024 {
            skip_count = U2048::ONE;
        } else {
            let mask = U2048::ONE.shl_vartime(bits).wrapping_sub(&U2048::ONE);
            let begin = forbidden_low
                .wrapping_sub(&first_wide)
                .wrapping_add(&mask)
                .shr_vartime(bits);
            let end = forbidden_high.wrapping_sub(&first_wide).shr_vartime(bits);
            if begin <= end {
                skip_start = begin;
                skip_count = end.wrapping_sub(&begin).wrapping_add(&U2048::ONE);
            }
        }
    }
    let eligible: U1024 = total.wrapping_sub(&skip_count).resize();
    if eligible == U1024::ZERO {
        task.zeroize();
        return Ok(false);
    }
    let cursor = sample(config, task.id, CandidateDomain::RsaRangeStart, &eligible)
        .ok_or("RSA range sampling failed")?;
    task.first = first;
    task.count = eligible.to_be_bytes();
    task.remaining = task.count;
    task.skip_start = skip_start.resize::<{ U1024::LIMBS }>().to_be_bytes();
    task.skip_count = skip_count.resize::<{ U1024::LIMBS }>().to_be_bytes();
    task.cursor = cursor.to_be_bytes();
    task.state = 2;
    task.winner = 0;
    Ok(true)
}

/// A launch interleaves q work across the compact list of active factors.
/// Return None for the unused tail of a short range without evaluating it.
pub fn q_at(config: &SearchConfig, task: &Task, offset: u32) -> Option<[u8; 128]> {
    if task.state != 2 {
        return None;
    }
    let offset = U1024::from_u32(offset);
    if offset >= U1024::from_be_slice(&task.remaining) {
        return None;
    }
    let count: U2048 = U1024::from_be_slice(&task.count).resize();
    let mut index: U2048 = U1024::from_be_slice(&task.cursor)
        .resize::<{ U2048::LIMBS }>()
        .wrapping_add(&offset.resize());
    if index >= count {
        index = index.wrapping_sub(&count);
    }
    if index >= U1024::from_be_slice(&task.skip_start).resize() {
        index = index.wrapping_add(&U1024::from_be_slice(&task.skip_count).resize());
    }
    let step = if config.suffix_bits >= 1024 {
        U2048::ZERO
    } else {
        index.shl_vartime(config.suffix_bits as usize)
    };
    let q = U1024::from_be_slice(&task.first)
        .resize::<{ U2048::LIMBS }>()
        .wrapping_add(&step);
    if q.bits_vartime() != 1024 {
        return None;
    }
    Some(q.resize::<{ U1024::LIMBS }>().to_be_bytes())
}

pub fn finish_tile(task: &mut Task, assigned: u32) {
    if task.winner != 0 {
        task.zeroize();
        return;
    }
    let remaining = U1024::from_be_slice(&task.remaining);
    let used = U1024::from_u32(assigned).min(remaining);
    if used == remaining {
        task.zeroize();
        return;
    }
    let count: U2048 = U1024::from_be_slice(&task.count).resize();
    let mut cursor: U2048 = U1024::from_be_slice(&task.cursor)
        .resize::<{ U2048::LIMBS }>()
        .wrapping_add(&used.resize());
    if cursor >= count {
        cursor = cursor.wrapping_sub(&count);
    }
    task.cursor = cursor.resize::<{ U1024::LIMBS }>().to_be_bytes();
    task.remaining = remaining.wrapping_sub(&used).to_be_bytes();
}

pub fn eligible_pair(
    p: &[u8; 128],
    q: &[u8; 128],
    pattern: &crate::search::hex_pattern::HexPattern,
) -> bool {
    let p = U1024::from_be_slice(p);
    let q = U1024::from_be_slice(q);
    let distance = if p > q {
        p.wrapping_sub(&q)
    } else {
        q.wrapping_sub(&p)
    };
    let n: U2048 = p.mul(&q);
    distance > U1024::ONE.shl_vartime(924)
        && n.bits_vartime() == 2048
        && pattern.matches(&n.to_be_bytes())
        && crate::crypto::rsa_prime::probable_prime(&q)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn records_are_padding_free() {
        assert_eq!(core::mem::size_of::<SearchConfig>(), 1072);
        assert_eq!(core::mem::size_of::<Task>(), 912);
        assert_eq!(core::mem::size_of::<Pair>(), 264);
        assert_eq!(core::mem::size_of::<Counts>(), 32);
    }
}
