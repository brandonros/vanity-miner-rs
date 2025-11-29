use crate::common::GlobalStats;
use crate::common::SharedBestHash;
use std::error::Error;
use std::sync::{Arc, RwLock};

pub mod cpu {
    use super::*;
    use rand::Rng as _;

    fn worker(
        thread_id: usize,
        username: String,
        shared_best_hash: Arc<RwLock<SharedBestHash>>,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut rng = rand::thread_rng();

        println!("[CPU-{}] Starting CPU shallenge worker thread", thread_id);

        loop {
            let rng_seed: u64 = rng.r#gen();

            // Get the current best hash (with minimal lock time)
            let current_target = {
                let best_hash_guard = shared_best_hash.read().unwrap();
                best_hash_guard.get_current()
            };

            // Create the request with the current best target
            let request = logic::ShallengeRequest {
                username: username.as_bytes(),
                username_len: username.len(),
                target_hash: &current_target,
                thread_idx: thread_id,
                rng_seed,
            };

            let result = logic::generate_and_check_shallenge(&request);

            global_stats.add_launch(1);

            if result.is_better {
                let nonce_string =
                    std::str::from_utf8(&result.nonce[0..result.nonce_len]).unwrap_or("invalid_utf8");

                // Try to update the global best hash
                let was_global_best = {
                    let mut best_hash_guard = shared_best_hash.write().unwrap();
                    best_hash_guard.update_if_better(result.hash)
                };

                if was_global_best {
                    println!("[CPU-{}] NEW GLOBAL BEST found: thread_idx = {}", thread_id, thread_id);
                    println!("[CPU-{}] NEW GLOBAL BEST hash: {}", thread_id, hex::encode(result.hash));
                    println!("[CPU-{}] NEW GLOBAL BEST nonce: {}", thread_id, nonce_string);
                    println!("[CPU-{}] Challenge string: {}/{}", thread_id, username, nonce_string);

                    global_stats.add_matches(1);
                    global_stats.print_stats(thread_id, 1);
                }
            }
        }
    }

    pub fn run(
        num_threads: usize,
        username: String,
        target_hash: Vec<u8>,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Starting CPU shallenge mode with {} threads", num_threads);

        // Convert Vec<u8> to [u8; 32] for the initial target
        let mut initial_target = [0u8; 32];
        initial_target.copy_from_slice(&target_hash);

        // Create shared state for the best hash found so far
        let shared_best_hash = Arc::new(RwLock::new(SharedBestHash::new(initial_target)));

        let mut handles = Vec::new();

        for i in 0..num_threads {
            let username_clone = username.clone();
            let shared_best_hash_clone = Arc::clone(&shared_best_hash);
            let stats_clone = Arc::clone(&global_stats);

            handles.push(std::thread::spawn(move || {
                worker(i, username_clone, shared_best_hash_clone, stats_clone)
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
        username: String,
        shared_best_hash: Arc<RwLock<SharedBestHash>>,
        module: &Module,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let username_bytes = username.as_bytes();
        let username_len: usize = username_bytes.len();

        let gpu = GpuContext::new(ordinal)?;
        let kernel = module.get_function("kernel_find_better_shallenge_nonce")?;
        gpu.print_launch_info(ordinal, "shallenge");

        let mut rng = rand::thread_rng();

        loop {
            let rng_seed: u64 = rng.r#gen::<u64>();

            // Get the current best hash (with minimal lock time)
            let current_target = {
                let best_hash_guard = shared_best_hash.read().unwrap();
                best_hash_guard.get_current()
            };

            let mut found_matches_slice = [0u32; 1];
            let mut found_hash = [0u8; 32];
            let mut found_nonce = [0u8; 64];
            let mut found_nonce_len = [0usize; 1];
            let mut found_thread_idx_slice = [0u32; 1];

            let username_dev = username_bytes.as_dbuf()?;
            let target_hash_dev = current_target.as_dbuf()?;
            let found_matches_slice_dev = found_matches_slice.as_dbuf()?;
            let found_hash_dev = found_hash.as_dbuf()?;
            let found_nonce_dev = found_nonce.as_dbuf()?;
            let found_nonce_len_dev = found_nonce_len.as_dbuf()?;
            let found_thread_idx_slice_dev = found_thread_idx_slice.as_dbuf()?;

            unsafe {
                launch!(
                    kernel<<<gpu.blocks_per_grid as u32, gpu.threads_per_block as u32, 0, gpu.stream>>>(
                        username_dev.as_device_ptr(),
                        username_len,
                        target_hash_dev.as_device_ptr(),
                        rng_seed,
                        found_matches_slice_dev.as_device_ptr(),
                        found_hash_dev.as_device_ptr(),
                        found_nonce_dev.as_device_ptr(),
                        found_nonce_len_dev.as_device_ptr(),
                        found_thread_idx_slice_dev.as_device_ptr(),
                    )
                )?;
            }

            gpu.stream.synchronize()?;
            global_stats.add_launch(gpu.operations_per_launch);

            found_matches_slice_dev.copy_to(&mut found_matches_slice)?;

            let found_matches = found_matches_slice[0];
            if found_matches != 0 {
                found_hash_dev.copy_to(&mut found_hash)?;
                found_nonce_dev.copy_to(&mut found_nonce)?;
                found_nonce_len_dev.copy_to(&mut found_nonce_len)?;
                found_thread_idx_slice_dev.copy_to(&mut found_thread_idx_slice)?;

                let found_thread_idx = found_thread_idx_slice[0];
                let nonce_len = found_nonce_len[0];
                let nonce_string = String::from_utf8(found_nonce[..nonce_len].to_vec()).unwrap();

                // Try to update the global best hash
                let was_global_best = {
                    let mut best_hash_guard = shared_best_hash.write().unwrap();
                    best_hash_guard.update_if_better(found_hash)
                };

                if was_global_best {
                    println!("[{ordinal}] NEW GLOBAL BEST found: seed = {rng_seed} thread_idx = {found_thread_idx}");
                    println!("[{ordinal}] NEW GLOBAL BEST hash: {}", hex::encode(found_hash));
                    println!("[{ordinal}] NEW GLOBAL BEST nonce: {}", nonce_string);
                    println!("[{ordinal}] Challenge string: {}/{}", username, nonce_string);

                    global_stats.add_matches(found_matches as usize);
                    global_stats.print_stats(ordinal, found_matches as u32);
                }
            }
        }
    }
}
