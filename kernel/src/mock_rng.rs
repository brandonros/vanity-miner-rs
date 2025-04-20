use gpu_rand::xoroshiro::rand_core::SeedableRng;
use gpu_rand::xoroshiro::rand_core::RngCore;

pub struct MockXoroshiro128StarStar {
    // We'll just use a simple counter for our mock
    counter: u64,
}

impl MockXoroshiro128StarStar {
    // Constructor method
    pub fn new(seed: u64) -> Self {
        Self { counter: seed }
    }
}

// Implement the SeedableRng trait
impl SeedableRng for MockXoroshiro128StarStar {
    type Seed = [u8; 16];

    // Initialize from a seed
    fn from_seed(seed: Self::Seed) -> Self {
        // Convert the first 8 bytes to a u64 for our counter initialization
        let mut seed_value: u64 = 0;
        for i in 0..8 {
            seed_value = (seed_value << 8) | seed[i] as u64;
        }
        Self { counter: seed_value }
    }

    // Create from a u64 seed
    fn seed_from_u64(seed: u64) -> Self {
        Self { counter: seed }
    }
}

// Implement the RngCore trait
impl RngCore for MockXoroshiro128StarStar {
    // Get the next u32 value (just return the lower 32 bits of our counter)
    fn next_u32(&mut self) -> u32 {
        let result = self.counter as u32;
        self.counter += 1;
        result
    }

    // Get the next u64 value (simple counter increment)
    fn next_u64(&mut self) -> u64 {
        let result = self.counter;
        self.counter += 1;
        result
    }

    // Fill a buffer with "random" bytes (just sequential values)
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for byte in dest.iter_mut() {
            *byte = self.counter as u8;
            self.counter += 1;
        }
    }
}