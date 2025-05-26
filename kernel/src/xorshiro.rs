pub fn generate_random_private_key(thread_idx: usize, rng_seed: u64) -> [u8; 32] {
    
    /*// Initialize state with thread_idx and seed
    let mut s0 = thread_idx as u64 ^ rng_seed;
    let mut s1 = s0.wrapping_add(0x9E3779B97F4A7C15); // Use golden ratio for second part of state
    
    if s0 == 0 && s1 == 0 {
        s0 = 1; // Avoid all-zero state
    }
    
    let mut private_key = [0u8; 32];
    
    for i in (0..32).step_by(8) {
        // Xoroshiro128** algorithm
        let result = s0.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        
        let bytes = [
            (result >> 56) as u8,
            (result >> 48) as u8,
            (result >> 40) as u8,
            (result >> 32) as u8,
            (result >> 24) as u8,
            (result >> 16) as u8,
            (result >> 8) as u8,
            result as u8
        ];
        
        // State update
        let s1_new = s1 ^ s0;
        s0 = s0.rotate_left(24) ^ s1_new ^ (s1_new << 16);
        s1 = s1_new.rotate_left(37);
        
        let end = core::cmp::min(i + 8, 32);
        private_key[i..end].copy_from_slice(&bytes[0..(end - i)]);
    }
    
    private_key*/

    use rand_core::{SeedableRng, RngCore};
    use rand_xoshiro::Xoroshiro128StarStar;
    let mut private_key = [0u8; 32];
    let mut rng = Xoroshiro128StarStar::seed_from_u64(rng_seed ^ (thread_idx as u64));
    rng.fill_bytes(&mut private_key);
    private_key
}
