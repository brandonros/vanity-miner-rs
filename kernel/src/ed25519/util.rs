#[macro_export]
macro_rules! load_8u {
    ($s:expr, $offset:expr) => {
        ($s[$offset] as u64)
            | (($s[$offset + 1] as u64) << 8)
            | (($s[$offset + 2] as u64) << 16)
            | (($s[$offset + 3] as u64) << 24)
            | (($s[$offset + 4] as u64) << 32)
            | (($s[$offset + 5] as u64) << 40)
            | (($s[$offset + 6] as u64) << 48)
            | (($s[$offset + 7] as u64) << 56)
    };
}