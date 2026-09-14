//! CPU oracle for the actual nonce function; JSON is consumed by the GPU runner.
fn main() {
    // Preserve the smallest independently recorded historical mismatch.
    let mut diagnostic = [0u8; 2];
    logic::generate_base64_nonce(0, 0, &mut diagnostic);
    assert_eq!(&diagnostic, b"pe", "seed-0 two-byte nonce oracle changed");
    let mut rows = Vec::new();
    for seed in [0u64, 1, 0xffff_ffff, 0x1_0000_0000, 0x8000_0000_0000_0000, u64::MAX] {
        for index in [0usize, 1, 31, 32, 257] {
            for length in [0usize, 1, 2, 7, 31, 32, 63, 64] {
                let mut bytes = [0xa5u8; 64];
                logic::generate_base64_nonce(index, seed, &mut bytes[..length]);
                let words: Vec<u64> = bytes.chunks_exact(8)
                    .map(|x| u64::from_le_bytes(x.try_into().unwrap())).collect();
                rows.push(format!("{{\"entry\":\"kernel_repro_nonce_sequence\",\"input\":[{index},{seed},{length}],\"expected\":{words:?}}}"));
            }
        }
    }
    println!("[{}]", rows.join(",\n"));
}
