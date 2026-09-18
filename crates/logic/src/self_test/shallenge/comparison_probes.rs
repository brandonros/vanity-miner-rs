//! Comparison probes owned by the shallenge self-test kernel.
use crate::self_test::black_box;

register_self_test! {
    /// compare hashes differing only at last byte
    fn compare_hashes_last_byte() -> u32 {
        let a = black_box([0x42; 32]);
        let mut b = a;
        b[31] = black_box(0x43);
        u32::from(super::compare_hashes(&a, &b) == -1 && super::compare_hashes(&b, &a) == 1)
    }
}
