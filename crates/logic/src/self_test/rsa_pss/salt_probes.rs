//! Focused salt probes for the rsa_pss self-test.
use crate::self_test::black_box;

register_self_test! {
    /// salt counter last value and exhaustion
    fn salt_counter_exhaustion() -> u32 {
        use crate::search::{message_window::WindowError, salt_counter::write_salt_counter};
        let base = black_box([173]);
        let mut out = black_box([0xa5]);
        if write_salt_counter(&base, black_box(255), &mut out) != Ok(()) || out != [172] {
            return 0;
        }
        out = black_box([0xa5]);
        u32::from(
            write_salt_counter(&base, black_box(256), &mut out) == Err(WindowError::Exhausted)
                && out == [0xa5],
        )
    }
}

register_self_test! {
    /// empty salt capacity and mismatched output length
    fn salt_empty_and_invalid() -> u32 {
        use crate::search::{message_window::WindowError, salt_counter::write_salt_counter};
        let empty = &black_box([] as [u8; 0]);
        let mut out = [0xa5; 1];
        u32::from(
            write_salt_counter(empty, black_box(0), &mut []) == Ok(())
                && write_salt_counter(empty, black_box(1), &mut []) == Err(WindowError::Exhausted)
                && write_salt_counter(empty, black_box(0), &mut out) == Err(WindowError::InvalidBounds)
                && out == [0xa5],
        )
    }
}

register_self_test! {
    /// salt carry beyond counter width
    fn salt_carry_beyond_u64() -> u32 {
        use crate::search::salt_counter::write_salt_counter;
        let mut base = [0xff; 16];
        base[0] = 0x12;
        let mut out = black_box([0xa5; 16]);
        let mut expected = [0; 16];
        expected[0] = 0x13;
        u32::from(
            write_salt_counter(&black_box(base), black_box(1), &mut out) == Ok(()) && out == expected,
        )
    }
}
