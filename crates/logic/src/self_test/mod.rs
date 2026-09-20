//! On-device / on-CPU self-test: runs known-answer tests for every logic
//! primitive against externally-validated expected values, writing
//! pass(1)/fail(0) per check into the results buffer.
//!
//! Each slot has a dedicated `register_self_test!` function. GPU mode runs eight mode-specific
//! kernels, using generated result indices. CPU mode calls all checks in sequence.
//!
//! Keep known-answer inputs opaque before the operation under test. A barrier
//! around the final boolean is too late: the operation can already be folded.
//! `black_box` uses a private volatile read on NVPTX so direct Metal compilation
//! preserves opaque inputs without target-specific inline assembly. Check emitted
//! LLVM/AIR and GPU results to verify the operations survive optimization.

/// Keep known-answer inputs opaque before the operation under test.
///
/// The NVPTX producer uses a volatile read from initialized private storage.
/// This is an identity operation, not a synchronization or memory-ordering API.
#[inline(always)]
pub fn black_box<T>(value: T) -> T {
    #[cfg(target_arch = "nvptx64")]
    {
        volatile_identity(value)
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        core::hint::black_box(value)
    }
}

#[cfg(any(target_arch = "nvptx64", test))]
#[inline(always)]
fn volatile_identity<T>(value: T) -> T {
    use core::mem::{ManuallyDrop, MaybeUninit};
    let value = ManuallyDrop::new(value);
    let mut copy = MaybeUninit::<T>::uninit();
    let (from, to) = (
        (&raw const *value).cast::<MaybeUninit<u8>>(),
        copy.as_mut_ptr().cast::<MaybeUninit<u8>>(),
    );
    // One volatile read per byte. LLVM 22 turns a volatile read of a whole T
    // into a load of one integer as wide as T (i256, i8576), which llvm-metal's
    // integer profile rightly refuses; volatile byte reads cannot be merged.
    // SAFETY: both pointers cover size_of::<T>() bytes of distinct storage.
    // MaybeUninit<u8> admits padding. ManuallyDrop prevents dropping the
    // original; ownership moves to the returned value, a byte copy of it.
    unsafe {
        let mut i = 0;
        while i < core::mem::size_of::<T>() {
            to.add(i).write(core::ptr::read_volatile(from.add(i)));
            i += 1;
        }
        copy.assume_init()
    }
}

#[macro_use]
mod registration;
include!("registry.rs");

#[cfg(any(
    feature = "self_test_p256_public_key",
    feature = "self_test_p256_signature",
    feature = "self_test_rsa_pss",
    feature = "self_test_rsa_modulus"
))]
mod known_answers;

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
pub(super) fn bytes_eq_prefix(actual: &[u8; 64], expected: &[u8]) -> bool {
    let n = expected.len();
    if n > actual.len() {
        return false;
    }
    let mut i = 0;
    while i < n {
        if actual[i] != expected[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(any(
    feature = "self_test_p256_public_key",
    feature = "self_test_p256_signature",
    feature = "self_test_rsa_pss"
))]
fn record_candidate(
    h: &mut crate::crypto::sha256::Sha256,
    result: crate::search::candidate_result::CandidateResult,
) {
    h.update(result.status.to_le_bytes());
    h.update(result.bytes);
}

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
pub struct IdxProbe(pub [u64; 5]);

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
impl core::ops::Index<usize> for IdxProbe {
    type Output = u64;
    fn index(&self, i: usize) -> &u64 {
        &(self.0[i])
    }
}

#[cfg(any(feature = "self_test_solana", feature = "self_test_bitcoin"))]
impl core::ops::IndexMut<usize> for IdxProbe {
    fn index_mut(&mut self, i: usize) -> &mut u64 {
        &mut (self.0[i])
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn volatile_identity_preserves_values_alignment_and_single_ownership() {
        use core::cell::Cell;
        #[repr(align(64))]
        struct Aligned([u64; 8]);
        let data = core::array::from_fn(|i| 1u64 << (i * 8));
        assert_eq!(volatile_identity(Aligned(data)).0, data);
        let pointer = data.as_ptr();
        assert_eq!(volatile_identity(pointer), pointer);
        volatile_identity(());
        let dropped = Cell::new(0);
        struct Owned<'a>(&'a Cell<u32>);
        impl Drop for Owned<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let owned = volatile_identity(Owned(&dropped));
        assert_eq!(dropped.get(), 0);
        drop(owned);
        assert_eq!(dropped.get(), 1);
    }

    #[test]
    fn enabled_self_test_checks_pass_on_cpu_and_disabled_slots_are_untouched() {
        const UNWRITTEN: u32 = 0xa5a5a5a5;
        let mut results = [UNWRITTEN; SELF_TEST_NUM_CHECKS];
        run_self_test(&mut results);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(
                r,
                if metadata::CASES[i].enabled {
                    1
                } else {
                    UNWRITTEN
                },
                "self-test check {} ({}) failed",
                i,
                metadata::CASES[i].name
            );
        }
    }
}
