//! Focused window probes for the p256_signature self-test.
use crate::self_test::black_box;

register_self_test! {
    /// message counter last value and exhaustion
    fn message_counter_exhaustion() -> u32 {
        use crate::search::message_window::{WindowError, hash_message_counter, write_message_counter};
        let mut message = black_box([0xa5; 4]);
        if write_message_counter(&mut message, black_box(1), black_box(1), black_box(255)) != Ok(())
            || message != [0xa5, 255, 0xa5, 0xa5]
        {
            return 0;
        }
        u32::from(
            write_message_counter(&mut message, black_box(1), black_box(1), black_box(256))
                == Err(WindowError::Exhausted)
                && message == [0xa5, 255, 0xa5, 0xa5]
                && hash_message_counter(&message, black_box(1), black_box(1), black_box(256))
                    == Err(WindowError::Exhausted),
        )
    }
}

register_self_test! {
    /// message window rejects invalid bounds without mutation
    fn message_window_invalid() -> u32 {
        use crate::search::message_window::{WindowError, hash_message_counter, write_message_counter};
        for (offset, length) in [(0, 0), (4, 1), (usize::MAX, 2)] {
            let mut message = black_box([0xa5; 4]);
            let offset = black_box(offset);
            let length = black_box(length);
            if write_message_counter(&mut message, offset, length, black_box(0))
                != Err(WindowError::InvalidBounds)
                || message != [0xa5; 4]
                || hash_message_counter(&message, offset, length, black_box(0))
                    != Err(WindowError::InvalidBounds)
            {
                return 0;
            }
        }
        1
    }
}

register_self_test! {
    /// message window across multiple sha256 blocks
    fn message_window_multiblock() -> u32 {
        use crate::search::message_window::{hash_message_counter, write_message_counter};
        let message = black_box([0xa5; 160]);
        let counter = black_box(0x0102030405060708090a0b0c0d0e0f10u128);
        let mut materialized = message;
        if write_message_counter(&mut materialized, black_box(7), black_box(129), counter) != Ok(()) {
            return 0;
        }
        // Independently specified bytes, including unchanged prefix/suffix and all zero padding.
        let mut expected = [0xa5; 160];
        expected[7..120].fill(0);
        expected[120..136].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        u32::from(
            materialized == expected
                && hash_message_counter(&message, black_box(7), black_box(129), counter)
                    == Ok(super::window_fixtures::LONG_WINDOW_SHA256),
        )
    }
}
