//! CPU oracle for the actual nonce function; JSON is consumed by the GPU runner.
fn main() {
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
    // Independent expected alphabet/device bytes, not the helper being tested.
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = [0x00u8, 0x7f, 0x80, 0xff, 0x11, 0x42, 0xa5, 0x5a];
    let packed = u64::from_le_bytes(bytes);
    for index in (0..128usize).chain([usize::MAX]) {
        rows.push(format!("{{\"entry\":\"kernel_repro_alphabet_helper\",\"input\":[{index},{packed}],\"expected\":[{},{}]}}", alphabet[index & 63], bytes[index & 7]));
    }
    println!("[{}]", rows.join(",\n"));
}
