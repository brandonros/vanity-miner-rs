//! Batch seed probes owned by the solana self-test kernel.
use core::hint::black_box;

register_self_test! {
    /// batch seed lane boundary and seed wrap
    fn batch_seed_boundary() -> u32 {
        use crate::search::xoroshiro::BatchSeed;
        let seed = black_box(BatchSeed {
            seed: u64::MAX,
            width: 32,
        });
        u32::from(
            seed.position(black_box(31)) == Some((u64::MAX, 31))
                && seed.position(black_box(32)) == Some((0, 0))
                && seed.position(black_box(33)) == Some((0, 1))
                && seed.position(black_box(65)) == Some((1, 1)),
        )
    }
}

register_self_test! {
    /// batch seed invalid and maximum widths
    fn batch_seed_invalid_width() -> u32 {
        use crate::search::xoroshiro::BatchSeed;
        let zero = black_box(BatchSeed { seed: 7, width: 0 });
        let large = black_box(BatchSeed {
            seed: 7,
            width: u32::MAX as u64 + 1,
        });
        let max = black_box(BatchSeed {
            seed: 7,
            width: u32::MAX as u64,
        });
        u32::from(
            zero.position(black_box(0)).is_none()
                && large.position(black_box(0)).is_none()
                && max.position(black_box(u32::MAX as u64)) == Some((8, 0)),
        )
    }
}
