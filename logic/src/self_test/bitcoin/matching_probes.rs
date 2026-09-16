//! Matching probes owned by the bitcoin self-test kernel.
use core::hint::black_box;

register_self_test! {
    /// byte pattern prefix suffix and late mismatches
    fn vanity_prefix_suffix() -> u32 {
        use crate::search::vanity::check_vanity_match;
        let data = black_box(*b"abcdef");
        u32::from(
            check_vanity_match(&data, black_box(b""), black_box(b"ef"))
                && check_vanity_match(&data, black_box(b"abc"), black_box(b"def"))
                && check_vanity_match(&data, black_box(b"abcd"), black_box(b"cdef"))
                && !check_vanity_match(&data, black_box(b"abd"), black_box(b"ef"))
                && !check_vanity_match(&data, black_box(b"abc"), black_box(b"deg")),
        )
    }
}

register_self_test! {
    /// byte pattern empty exact and overlong inputs
    fn vanity_length_boundaries() -> u32 {
        use crate::search::vanity::check_vanity_match;
        let data = black_box(*b"abc");
        u32::from(
            check_vanity_match(&data, black_box(b"abc"), black_box(b"abc"))
                && check_vanity_match(black_box(b""), black_box(b""), black_box(b""))
                && !check_vanity_match(&data, black_box(b"abcd"), black_box(b""))
                && !check_vanity_match(&data, black_box(b""), black_box(b"abcd"))
                && !check_vanity_match(black_box(b""), black_box(b"a"), black_box(b"")),
        )
    }
}
