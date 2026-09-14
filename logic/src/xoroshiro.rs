use rand_core::{SeedableRng, RngCore};
use rand_xoshiro::Xoroshiro128StarStar;

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15u64);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9u64);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111ebu64);
    x ^ (x >> 31)
}

pub fn generate_random_private_key(thread_idx: usize, rng_seed: u64) -> [u8; 32] {
    let mixed_seed = splitmix64(rng_seed.wrapping_add(thread_idx as u64));
    let mut private_key = [0u8; 32];
    let mut rng = Xoroshiro128StarStar::seed_from_u64(mixed_seed);
    rng.fill_bytes(&mut private_key);
    private_key
}

pub fn generate_base64_nonce(thread_idx: usize, rng_seed: u64, nonce: &mut [u8]) {
    let mixed_seed = splitmix64(rng_seed.wrapping_add(thread_idx as u64));
    let mut rng = Xoroshiro128StarStar::seed_from_u64(mixed_seed);
    for byte in nonce.iter_mut() {
        let idx = (rng.next_u32() % 64) as usize;
        *byte = BASE64_CHARS[idx];
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn should_generate_private_key_correctly() {
        let priv_key = generate_random_private_key(3, 583437459223573146);
        let expected: [u8; 32] = hex::decode("fa9ce9b02dc28a48f7e9d15506d3d2c443d596565fa05214b0ff7c5ab5e7956b")
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(priv_key, expected);
    }

    #[test]
    fn should_generate_base64_nonce_in_alphabet_and_deterministically() {
        let mut a = [0u8; 21];
        let mut b = [0u8; 21];
        generate_base64_nonce(7, 12345, &mut a);
        generate_base64_nonce(7, 12345, &mut b);
        assert_eq!(a, b, "same seed must yield same nonce");
        for &byte in &a {
            assert!(BASE64_CHARS.contains(&byte), "byte {:#x} not in base64 alphabet", byte);
        }
    }
}

// Same alphabet as the production nonce generator. Keep the two pointer origins
// in a shared helper so generated PTX can expose helper address-space mistakes.
#[cfg(feature = "self_test")]
#[inline(never)]
unsafe fn repro_read_byte(base: *const u8, index: usize) -> u8 {
    unsafe { *base.add(index) }
}

#[cfg(feature = "self_test")]
pub fn repro_alphabet_helper(index: usize, bytes: &[u8; 8]) -> (u64, u64) {
    // Both offsets are in bounds by construction. Isolate the pointer read,
    // not an additional bounds-check/trap control-flow path in the helper.
    // Keep the dynamic index inside the helper: a plain dereference can be
    // promoted to a byte argument even when the function is not inlined.
    unsafe {
        (
            repro_read_byte(BASE64_CHARS.as_ptr(), index & 63) as u64,
            repro_read_byte(bytes.as_ptr(), index & 7) as u64,
        )
    }
}
