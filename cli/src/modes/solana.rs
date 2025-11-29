use crate::common::GlobalStats;
use std::error::Error;
use std::sync::Arc;

pub mod cpu {
    use super::*;
    use rand::Rng as _;

    fn worker(
        thread_id: usize,
        prefix_bytes: Vec<u8>,
        suffix_bytes: Vec<u8>,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut rng = rand::thread_rng();

        println!("[CPU-{thread_id}] Starting CPU solana vanity worker thread");

        loop {
            let rng_seed: u64 = rng.r#gen();

            let request = logic::SolanaVanityKeyRequest {
                prefix: &prefix_bytes,
                suffix: &suffix_bytes,
                thread_idx: thread_id,
                rng_seed,
            };

            let result = logic::generate_and_check_solana_vanity_key(&request);

            global_stats.add_launch(1);

            if result.matches {
                let encoded_str =
                    std::str::from_utf8(&result.encoded_public_key[0..result.encoded_len])
                        .unwrap_or("invalid_utf8");

                println!("[CPU-{thread_id}] Vanity match: rng_seed = {rng_seed}");
                println!("[CPU-{thread_id}] Vanity match: thread_idx = {thread_id}");
                println!("[CPU-{thread_id}] Vanity match: encoded_public_key = {encoded_str}");
                println!(
                    "[CPU-{thread_id}] Vanity match: public_key = {}",
                    hex::encode(result.public_key)
                );
                println!(
                    "[CPU-{thread_id}] Vanity match: private_key = {}",
                    hex::encode(result.private_key)
                );
                println!(
                    "[CPU-{thread_id}] Vanity match: wallet = {}",
                    hex::encode([result.private_key, result.public_key].concat())
                );

                global_stats.add_matches(1);
                global_stats.print_stats(thread_id, 1);
            }
        }
    }

    pub fn run(
        num_threads: usize,
        prefix: String,
        suffix: String,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let prefix_bytes = prefix.as_bytes().to_vec();
        let suffix_bytes = suffix.as_bytes().to_vec();

        println!("Starting CPU solana vanity mode with {} threads", num_threads);

        let mut handles = Vec::new();

        for i in 0..num_threads {
            let prefix_clone = prefix_bytes.clone();
            let suffix_clone = suffix_bytes.clone();
            let stats_clone = Arc::clone(&global_stats);

            handles.push(std::thread::spawn(move || {
                worker(i, prefix_clone, suffix_clone, stats_clone)
            }));
        }

        for handle in handles {
            handle.join().unwrap()?;
        }

        Ok(())
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
        prefix: String,
        suffix: String,
        module: &Module,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let prefix_bytes = prefix.as_bytes().to_vec();
        let suffix_bytes = suffix.as_bytes().to_vec();

        let gpu = GpuContext::new(ordinal)?;
        let kernel = module.get_function("kernel_find_solana_vanity_private_key")?;
        gpu.print_launch_info(ordinal, "solana vanity");

        let mut rng = rand::thread_rng();

        loop {
            let rng_seed: u64 = rng.r#gen::<u64>();

            let mut found_matches_slice = [0u32; 1];
            let mut found_private_key = [0u8; 32];
            let mut found_public_key = [0u8; 32];
            let mut found_encoded_public_key = [0u8; 64];
            let mut found_thread_idx_slice = [0u32; 1];

            let prefix_dev = prefix_bytes.as_dbuf()?;
            let suffix_dev = suffix_bytes.as_dbuf()?;
            let found_matches_dev = found_matches_slice.as_dbuf()?;
            let found_private_key_dev = found_private_key.as_dbuf()?;
            let found_public_key_dev = found_public_key.as_dbuf()?;
            let found_encoded_public_key_dev = found_encoded_public_key.as_dbuf()?;
            let found_thread_idx_dev = found_thread_idx_slice.as_dbuf()?;

            unsafe {
                launch!(
                    kernel<<<gpu.blocks_per_grid as u32, gpu.threads_per_block as u32, 0, gpu.stream>>>(
                        prefix_dev.as_device_ptr(),
                        prefix_bytes.len(),
                        suffix_dev.as_device_ptr(),
                        suffix_bytes.len(),
                        rng_seed,
                        found_matches_dev.as_device_ptr(),
                        found_private_key_dev.as_device_ptr(),
                        found_public_key_dev.as_device_ptr(),
                        found_encoded_public_key_dev.as_device_ptr(),
                        found_thread_idx_dev.as_device_ptr(),
                    )
                )?;
            }

            gpu.stream.synchronize()?;
            global_stats.add_launch(gpu.operations_per_launch);

            found_matches_dev.copy_to(&mut found_matches_slice)?;

            if found_matches_slice[0] != 0 {
                found_private_key_dev.copy_to(&mut found_private_key)?;
                found_public_key_dev.copy_to(&mut found_public_key)?;
                found_encoded_public_key_dev.copy_to(&mut found_encoded_public_key)?;
                found_thread_idx_dev.copy_to(&mut found_thread_idx_slice)?;

                let found_thread_idx = found_thread_idx_slice[0];
                let found_encoded_public_key_string =
                    String::from_utf8(found_encoded_public_key.to_vec())
                        .unwrap_or_else(|_| "invalid_utf8".to_string());

                println!("[{ordinal}] Vanity match: seed = {rng_seed} thread_idx = {found_thread_idx}");
                println!("[{ordinal}] Vanity match: encoded_public_key = {found_encoded_public_key_string}");
                println!(
                    "[{ordinal}] Vanity match: public_key = {}",
                    hex::encode(found_public_key)
                );
                println!(
                    "[{ordinal}] Vanity match: private_key = {}",
                    hex::encode(found_private_key)
                );
                println!(
                    "[{ordinal}] Vanity match: wallet = {}",
                    hex::encode([found_private_key, found_public_key].concat())
                );

                global_stats.add_matches(found_matches_slice[0] as usize);
                global_stats.print_stats(ordinal, found_matches_slice[0]);
            }
        }
    }
}
