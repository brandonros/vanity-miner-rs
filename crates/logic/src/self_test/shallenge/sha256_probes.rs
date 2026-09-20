//! Sha256 probes owned by the shallenge self-test kernel.
use crate::self_test::black_box;

use super::sha256_fixtures::*;

register_self_test! {
    /// sha256 padding length 0
    fn sha256_padding_0() -> u32 {
        // Expected digest generated independently with Python hashlib.sha256(bytes(range(0))).
        let input = black_box([0u8; 0]);
        u32::from(
            crate::crypto::sha256::sha256_from_bytes(input.as_slice()) == SHA256_0_EXPECTED,
        )
    }
}

register_self_test! {
    /// sha256 padding length 55
    fn sha256_padding_55() -> u32 {
        // Expected digest generated independently with Python hashlib.sha256(bytes(range(55))).
        let input = black_box(core::array::from_fn::<_, 55, _>(|i| i as u8));
        u32::from(
            crate::crypto::sha256::sha256_from_bytes(input.as_slice()) == SHA256_55_EXPECTED,
        )
    }
}

register_self_test! {
    /// sha256 padding length 56
    fn sha256_padding_56() -> u32 {
        // Expected digest generated independently with Python hashlib.sha256(bytes(range(56))).
        let input = black_box(core::array::from_fn::<_, 56, _>(|i| i as u8));
        u32::from(
            crate::crypto::sha256::sha256_from_bytes(input.as_slice()) == SHA256_56_EXPECTED,
        )
    }
}

register_self_test! {
    /// sha256 padding length 63
    fn sha256_padding_63() -> u32 {
        // Expected digest generated independently with Python hashlib.sha256(bytes(range(63))).
        let input = black_box(core::array::from_fn::<_, 63, _>(|i| i as u8));
        u32::from(
            crate::crypto::sha256::sha256_from_bytes(input.as_slice()) == SHA256_63_EXPECTED,
        )
    }
}

register_self_test! {
    /// sha256 padding length 64
    fn sha256_padding_64() -> u32 {
        // Expected digest generated independently with Python hashlib.sha256(bytes(range(64))).
        let input = black_box(core::array::from_fn::<_, 64, _>(|i| i as u8));
        u32::from(
            crate::crypto::sha256::sha256_from_bytes(input.as_slice()) == SHA256_64_EXPECTED,
        )
    }
}

register_self_test! {
    /// sha256 padding length 65
    fn sha256_padding_65() -> u32 {
        // Expected digest generated independently with Python hashlib.sha256(bytes(range(65))).
        let input = black_box(core::array::from_fn::<_, 65, _>(|i| i as u8));
        u32::from(
            crate::crypto::sha256::sha256_from_bytes(input.as_slice()) == SHA256_65_EXPECTED,
        )
    }
}

register_self_test! {
    /// sha256 split updates across block boundary
    fn sha256_streaming_boundary() -> u32 {
        // Same independent 65-byte vector, with buffered, empty, and block-completing updates.
        let input = black_box(core::array::from_fn::<_, 65, _>(|i| i as u8));
        for split in [1usize, 55, 56, 63, 64] {
            let split = black_box(split);
            if split > input.len() {
                return 0;
            }
            let mut hash = crate::crypto::sha256::Sha256::new();
            hash.update(&input[..split]);
            hash.update(&black_box([] as [u8; 0]));
            hash.update(&input[split..]);
            if hash.finalize() != SHA256_65_EXPECTED {
                return 0;
            }
        }
        1
    }
}

register_self_test! {
    /// sha256 four full input blocks and a separate padding block
    fn sha256_multiblock() -> u32 {
        let input = black_box(core::array::from_fn::<_, 256, _>(|i| i as u8));
        u32::from(crate::crypto::sha256::sha256_from_bytes(&input) == SHA256_256_EXPECTED)
    }
}

register_self_test! {
    /// sha256 repeated short updates and full blocks share one known answer
    fn sha256_streaming_chunks() -> u32 {
        let input = black_box(core::array::from_fn::<_, 256, _>(|i| i as u8));
        // Short updates must accumulate without prematurely compressing a block;
        // the larger sizes alternate between buffered and direct compression.
        for chunk_size in [1usize, 7, 63, 64, 65] {
            let mut hash = crate::crypto::sha256::Sha256::new();
            let chunk_size = black_box(chunk_size);
            if chunk_size == 0 {
                return 0;
            }
            for chunk in input.chunks(chunk_size) {
                hash.update(chunk);
                hash.update(&black_box([] as [u8; 0]));
            }
            if hash.finalize() != SHA256_256_EXPECTED {
                return 0;
            }
        }
        1
    }
}
