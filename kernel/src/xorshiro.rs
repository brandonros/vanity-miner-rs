use rand_core::{SeedableRng, RngCore};
use rand_xoshiro::Xoshiro256Plus;

pub fn generate_random_private_key(thread_idx: usize, rng_seed: u64) -> [u8; 32] {
    let mut private_key = [0u8; 32];
    let mut rng = Xoshiro256Plus::seed_from_u64(rng_seed ^ (thread_idx as u64));
    rng.fill_bytes(&mut private_key);
    private_key
}
