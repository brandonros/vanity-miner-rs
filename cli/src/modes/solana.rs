use common::GlobalStats;
use std::error::Error;
use std::sync::Arc;

pub mod cpu {
    use super::*;
    use rand::Rng as _;

    fn worker(
        thread_id: usize,
        vanity_prefix: String,
        vanity_suffix: String,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        let vanity_prefix_bytes = vanity_prefix.as_bytes();
        let vanity_suffix_bytes = vanity_suffix.as_bytes();

        println!("[CPU-{}] Starting CPU vanity worker thread", thread_id);

        loop {
            let rng_seed: u64 = rng.r#gen();

            let request = logic::SolanaVanityKeyRequest {
                prefix: vanity_prefix_bytes,
                suffix: vanity_suffix_bytes,
                thread_idx: thread_id,
                rng_seed,
            };

            let result = logic::generate_and_check_solana_vanity_key(&request);

            global_stats.add_launch(1);

            if result.matches {
                let encoded_str =
                    std::str::from_utf8(&result.encoded_public_key[0..result.encoded_len])
                        .unwrap_or("invalid_utf8");

                println!("[CPU-{}] Vanity match: rng_seed = {}", thread_id, rng_seed);
                println!("[CPU-{}] Vanity match: thread_idx = {}", thread_id, thread_id);
                println!("[CPU-{}] Vanity match: encoded_public_key = {}", thread_id, encoded_str);
                println!("[CPU-{}] Vanity match: public_key = {}", thread_id, hex::encode(result.public_key));
                println!("[CPU-{}] Vanity match: private_key = {}", thread_id, hex::encode(result.private_key));
                println!("[CPU-{}] Vanity match: wallet = {}", thread_id, hex::encode([result.private_key, result.public_key].concat()));

                global_stats.add_matches(1);
                global_stats.print_stats(thread_id, 1);
            }
        }
    }

    pub fn run(
        num_threads: usize,
        vanity_prefix: String,
        vanity_suffix: String,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Starting CPU vanity mode with {} threads", num_threads);

        let mut handles = Vec::new();

        for i in 0..num_threads {
            let vanity_prefix_clone = vanity_prefix.clone();
            let vanity_suffix_clone = vanity_suffix.clone();
            let stats_clone = Arc::clone(&global_stats);

            handles.push(std::thread::spawn(move || {
                worker(i, vanity_prefix_clone, vanity_suffix_clone, stats_clone)
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
    use cust::device::Device;
    use cust::launch;
    use cust::memory::CopyDestination;
    use cust::module::Module;
    use cust::prelude::Context;
    use cust::stream::{Stream, StreamFlags};
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

        let device = Device::get_device(ordinal as u32)?;
        let ctx = Context::new(device)?;
        cust::context::CurrentContext::set_current(&ctx)?;

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let find_solana_vanity_private_key =
            module.get_function("kernel_find_solana_vanity_private_key")?;

        let number_of_streaming_multiprocessors =
            device.get_attribute(cust::device::DeviceAttribute::MultiprocessorCount)? as usize;
        let blocks_per_sm = std::env::var("BLOCKS_PER_SM")
            .unwrap_or("128".to_string())
            .parse::<usize>()
            .unwrap();
        let threads_per_block = std::env::var("THREADS_PER_BLOCK")
            .unwrap_or("256".to_string())
            .parse::<usize>()
            .unwrap();
        let blocks_per_grid = number_of_streaming_multiprocessors * blocks_per_sm;
        let operations_per_launch = blocks_per_grid * threads_per_block;

        println!(
            "[{ordinal}] Starting vanity search loop ({} blocks per grid, {} threads per block, {} operations per launch)",
            blocks_per_grid, threads_per_block, operations_per_launch
        );

        let mut rng = rand::thread_rng();

        loop {
            let rng_seed: u64 = rng.r#gen::<u64>();

            let mut found_matches_slice = [0u32; 1];
            let mut found_private_key = [0u8; 32];
            let mut found_public_key = [0u8; 32];
            let mut found_encoded_public_key = [0u8; 64];
            let mut found_thread_idx_slice = [0u32; 1];

            let vanity_prefix_dev = vanity_prefix_bytes.as_dbuf()?;
            let vanity_suffix_dev = vanity_suffix_bytes.as_dbuf()?;
            let found_matches_slice_dev = found_matches_slice.as_dbuf()?;
            let found_private_key_dev = found_private_key.as_dbuf()?;
            let found_public_key_dev = found_public_key.as_dbuf()?;
            let found_encoded_public_key_dev = found_encoded_public_key.as_dbuf()?;
            let found_thread_idx_slice_dev = found_thread_idx_slice.as_dbuf()?;

            unsafe {
                launch!(
                    find_solana_vanity_private_key<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                        vanity_prefix_dev.as_device_ptr(),
                        vanity_prefix_len,
                        vanity_suffix_dev.as_device_ptr(),
                        vanity_suffix_len,
                        rng_seed,
                        found_matches_slice_dev.as_device_ptr(),
                        found_private_key_dev.as_device_ptr(),
                        found_public_key_dev.as_device_ptr(),
                        found_encoded_public_key_dev.as_device_ptr(),
                        found_thread_idx_slice_dev.as_device_ptr(),
                    )
                )?;
            }

            stream.synchronize()?;
            global_stats.add_launch(operations_per_launch);

            found_matches_slice_dev.copy_to(&mut found_matches_slice)?;

            let found_matches = found_matches_slice[0];
            if found_matches != 0 {
                found_private_key_dev.copy_to(&mut found_private_key)?;
                found_public_key_dev.copy_to(&mut found_public_key)?;
                found_encoded_public_key_dev.copy_to(&mut found_encoded_public_key)?;
                found_thread_idx_slice_dev.copy_to(&mut found_thread_idx_slice)?;

                let found_thread_idx = found_thread_idx_slice[0];
                let found_encoded_public_key_string =
                    String::from_utf8(found_encoded_public_key.to_vec()).unwrap();
                println!("[{ordinal}] Vanity match: seed = {rng_seed} thread_idx = {found_thread_idx}");
                println!("[{ordinal}] Vanity match: encoded_public_key = {found_encoded_public_key_string}");
                println!("[{ordinal}] Vanity match: public_key = {}", hex::encode(found_public_key));
                println!("[{ordinal}] Vanity match: private_key = {}", hex::encode(found_private_key));
                println!("[{ordinal}] Vanity match: wallet = {}", hex::encode([found_private_key, found_public_key].concat()));

                global_stats.add_matches(found_matches as usize);
                global_stats.print_stats(ordinal, found_matches as u32);
            }
        }
    }
}
