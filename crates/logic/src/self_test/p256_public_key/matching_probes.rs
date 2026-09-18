//! Matching probes owned by the p256_public_key self-test kernel.
use crate::self_test::black_box;

register_self_test! {
    /// hex pattern odd nibbles suffix and width
    fn hex_pattern_nibbles() -> u32 {
        use crate::search::hex_pattern::HexPattern;
        // Construction occurs on the host in production; black_box the resulting device record.
        let Ok(pattern) = HexPattern::new("AbC", "dEf", 4) else { return 0; };
        let pattern = black_box(pattern);
        u32::from(
            pattern.matches(&black_box([0xab, 0xc0, 0x0d, 0xef]))
                && pattern.matches(&black_box([0xab, 0xcf, 0xfd, 0xef]))
                && !pattern.matches(&black_box([0xab, 0xbc, 0x0d, 0xef]))
                && !pattern.matches(&black_box([0xab, 0xc0, 0xd0, 0xef]))
                && !pattern.matches(&black_box([0xab, 0xc0, 0x0d, 0xef, 0])),
        )
    }
}

register_self_test! {
    /// hex pattern maximum width checks the last byte and ignores free nibbles
    fn hex_pattern_max_width() -> u32 {
        use crate::search::hex_pattern::HexPattern;
        // The matcher is shared with RSA's 256-byte targets.
        let Ok(pattern) = HexPattern::new("d", "f", 256) else { return 0; };
        let pattern = black_box(pattern);
        let mut input = black_box([0xa5; 256]);
        input[0] = black_box(0xda);
        input[255] = black_box(0x1f);
        if !pattern.matches(&input) || pattern.matches(&input[..255]) {
            return 0;
        }
        input[0] = black_box(0xd0);
        input[127] = black_box(0);
        input[255] = black_box(0xef);
        if !pattern.matches(&input) {
            return 0;
        }
        input[255] = black_box(0xee);
        if pattern.matches(&input) {
            return 0;
        }
        input[255] = black_box(0xef);
        input[0] = black_box(0xc0);
        u32::from(!pattern.matches(&input))
    }
}
