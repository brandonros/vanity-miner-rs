use crate::common::GlobalStats;
use crate::modes::VanityMode;
use std::error::Error;
use std::sync::Arc;

/// Bitcoin vanity address mode
pub struct BitcoinVanity;

impl VanityMode for BitcoinVanity {
    type Result = logic::BitcoinVanityKeyResult;

    const NAME: &'static str = "bitcoin vanity";

    fn generate_and_check(
        prefix: &[u8],
        suffix: &[u8],
        thread_idx: usize,
        rng_seed: u64,
    ) -> Self::Result {
        let request = logic::BitcoinVanityKeyRequest {
            prefix,
            suffix,
            thread_idx,
            rng_seed,
        };
        logic::generate_and_check_bitcoin_vanity_key(&request)
    }

    fn is_match(result: &Self::Result) -> bool {
        result.matches
    }

    fn print_match(thread_id: usize, rng_seed: u64, result: &Self::Result) {
        let encoded_public_key_str =
            std::str::from_utf8(&result.encoded_public_key[0..result.encoded_len])
                .unwrap_or("invalid_utf8");
        let mut encoded_private_key = [0u8; 64];
        let encoded_len =
            logic::private_key_to_wif(&result.private_key, true, false, &mut encoded_private_key);
        let encoded_private_key_str =
            std::str::from_utf8(&encoded_private_key[0..encoded_len]).unwrap_or("invalid_utf8");

        println!("[CPU-{thread_id}] Vanity match: rng_seed = {rng_seed}");
        println!("[CPU-{thread_id}] Vanity match: thread_idx = {thread_id}");
        println!("[CPU-{thread_id}] Vanity match: encoded_public_key = {encoded_public_key_str}");
        println!("[CPU-{thread_id}] Vanity match: public_key = {}", hex::encode(result.public_key));
        println!("[CPU-{thread_id}] Vanity match: public_key_hash = {}", hex::encode(result.public_key_hash));
        println!("[CPU-{thread_id}] Vanity match: private_key = {}", hex::encode(result.private_key));
        println!("[CPU-{thread_id}] Vanity match: wallet = {encoded_private_key_str}");
    }
}

pub mod cpu {
    use super::*;

    pub fn run(
        num_threads: usize,
        vanity_prefix: String,
        vanity_suffix: String,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        crate::modes::run_cpu_vanity::<BitcoinVanity>(num_threads, vanity_prefix, vanity_suffix, global_stats)
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use crate::common::GpuContext;
    use cust::launch;
    use cust::memory::CopyDestination;
    use cust::module::Module;
    use cust::util::SliceExt;
    use rand::Rng;

    pub fn run(
        ordinal: usize,
        vanity_prefix: String,
        vanity_suffix: String,
        module: &Module,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let vanity_prefix_bytes = vanity_prefix.as_bytes();
        let vanity_prefix_len: usize = vanity_prefix_bytes.len();
        let vanity_suffix_bytes = vanity_suffix.as_bytes();
        let vanity_suffix_len: usize = vanity_suffix_bytes.len();

        let gpu = GpuContext::new(ordinal)?;
        let kernel = module.get_function("kernel_find_bitcoin_vanity_private_key")?;
        gpu.print_launch_info(ordinal, "vanity");

        let mut rng = rand::thread_rng();

        loop {
            let rng_seed: u64 = rng.r#gen::<u64>();

            let mut found_matches_slice = [0u32; 1];
            let mut found_private_key = [0u8; 32];
            let mut found_public_key = [0u8; 33]; // Bitcoin compressed public key is 33 bytes
            let mut found_public_key_hash = [0u8; 20]; // Bitcoin address hash is 20 bytes
            let mut found_encoded_public_key = [0u8; 64]; // Bitcoin address encoding buffer
            let mut found_encoded_len_slice = [0u32; 1]; // Length of encoded address
            let mut found_thread_idx_slice = [0u32; 1];

            let vanity_prefix_dev = vanity_prefix_bytes.as_dbuf()?;
            let vanity_suffix_dev = vanity_suffix_bytes.as_dbuf()?;
            let found_matches_slice_dev = found_matches_slice.as_dbuf()?;
            let found_private_key_dev = found_private_key.as_dbuf()?;
            let found_public_key_dev = found_public_key.as_dbuf()?;
            let found_public_key_hash_dev = found_public_key_hash.as_dbuf()?;
            let found_encoded_public_key_dev = found_encoded_public_key.as_dbuf()?;
            let found_encoded_len_slice_dev = found_encoded_len_slice.as_dbuf()?;
            let found_thread_idx_slice_dev = found_thread_idx_slice.as_dbuf()?;

            unsafe {
                launch!(
                    kernel<<<gpu.blocks_per_grid as u32, gpu.threads_per_block as u32, 0, gpu.stream>>>(
                        vanity_prefix_dev.as_device_ptr(),
                        vanity_prefix_len,
                        vanity_suffix_dev.as_device_ptr(),
                        vanity_suffix_len,
                        rng_seed,
                        found_matches_slice_dev.as_device_ptr(),
                        found_private_key_dev.as_device_ptr(),
                        found_public_key_dev.as_device_ptr(),
                        found_public_key_hash_dev.as_device_ptr(),
                        found_encoded_public_key_dev.as_device_ptr(),
                        found_encoded_len_slice_dev.as_device_ptr(),
                        found_thread_idx_slice_dev.as_device_ptr(),
                    )
                )?;
            }

            gpu.stream.synchronize()?;
            global_stats.add_launch(gpu.operations_per_launch);

            found_matches_slice_dev.copy_to(&mut found_matches_slice)?;

            let found_matches = found_matches_slice[0];
            if found_matches != 0 {
                found_private_key_dev.copy_to(&mut found_private_key)?;
                found_public_key_dev.copy_to(&mut found_public_key)?;
                found_public_key_hash_dev.copy_to(&mut found_public_key_hash)?;
                found_encoded_public_key_dev.copy_to(&mut found_encoded_public_key)?;
                found_encoded_len_slice_dev.copy_to(&mut found_encoded_len_slice)?;
                found_thread_idx_slice_dev.copy_to(&mut found_thread_idx_slice)?;

                let found_thread_idx = found_thread_idx_slice[0];
                let encoded_len = found_encoded_len_slice[0] as usize;
                let encoded_public_key_str =
                    String::from_utf8(found_encoded_public_key[0..encoded_len].to_vec()).unwrap();

                // Generate WIF private key format
                let mut encoded_private_key = [0u8; 64];
                let encoded_private_key_len =
                    logic::private_key_to_wif(&found_private_key, true, false, &mut encoded_private_key);
                let encoded_private_key_str =
                    String::from_utf8(encoded_private_key[0..encoded_private_key_len].to_vec()).unwrap();

                println!("[{ordinal}] Vanity match: seed = {rng_seed} thread_idx = {found_thread_idx}");
                println!("[{ordinal}] Vanity match: encoded_public_key = {encoded_public_key_str}");
                println!("[{ordinal}] Vanity match: public_key = {}", hex::encode(found_public_key));
                println!("[{ordinal}] Vanity match: public_key_hash = {}", hex::encode(found_public_key_hash));
                println!("[{ordinal}] Vanity match: private_key = {}", hex::encode(found_private_key));
                println!("[{ordinal}] Vanity match: wallet = {encoded_private_key_str}");

                global_stats.add_matches(found_matches as usize);
                global_stats.print_stats(ordinal, found_matches as u32);
            }
        }
    }
}
