use crate::common::GlobalStats;
use crate::common::SharedBestHash;
use std::error::Error;
use std::sync::{Arc, RwLock};

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;
    use crate::common::spawn_cpu_workers;
    use rand::Rng as _;

    struct WorkerData {
        username: String,
        shared_best_hash: Arc<RwLock<SharedBestHash>>,
        global_stats: Arc<GlobalStats>,
    }

    fn worker(thread_id: usize, data: Arc<WorkerData>) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut rng = rand::thread_rng();

        println!("[CPU-{}] Starting CPU shallenge worker thread", thread_id);

        loop {
            let rng_seed: u64 = rng.r#gen();

            // Get the current best hash (with minimal lock time)
            // Use unwrap_or_else to recover data even if lock is poisoned
            let current_target = {
                let best_hash_guard = data.shared_best_hash.read().unwrap_or_else(|e| e.into_inner());
                best_hash_guard.get_current()
            };

            // Create the request with the current best target
            let request = logic::ShallengeRequest {
                username: data.username.as_bytes(),
                username_len: data.username.len(),
                target_hash: &current_target,
                thread_idx: thread_id,
                rng_seed,
            };

            let result = logic::generate_and_check_shallenge(&request);

            data.global_stats.add_launch(1);

            if result.is_better {
                let nonce_string =
                    std::str::from_utf8(&result.nonce[0..result.nonce_len]).unwrap_or("invalid_utf8");

                // Try to update the global best hash
                let was_global_best = {
                    let mut best_hash_guard = data.shared_best_hash.write().unwrap_or_else(|e| e.into_inner());
                    best_hash_guard.update_if_better(result.hash)
                };

                if was_global_best {
                    println!("[CPU-{}] NEW GLOBAL BEST found: thread_idx = {}", thread_id, thread_id);
                    println!("[CPU-{}] NEW GLOBAL BEST hash: {}", thread_id, hex::encode(result.hash));
                    println!("[CPU-{}] NEW GLOBAL BEST nonce: {}", thread_id, nonce_string);
                    println!("[CPU-{}] Challenge string: {}/{}", thread_id, data.username, nonce_string);

                    data.global_stats.add_matches(1);
                    data.global_stats.print_stats(thread_id, 1);
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

        let data = Arc::new(WorkerData {
            username,
            shared_best_hash,
            global_stats,
        });

        spawn_cpu_workers(num_threads, data, worker)
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use crate::common::GpuContext;
    use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
    use kernels::kernels::LoadedModule;
    use rand::Rng;

    pub fn run(
        ordinal: usize,
        username: String,
        shared_best_hash: Arc<RwLock<SharedBestHash>>,
        ctx: &Arc<CudaContext>,
        module: &LoadedModule,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let username_bytes = username.as_bytes();
        let username_len: usize = username_bytes.len();

        let gpu = GpuContext::new(ctx)?;
        gpu.print_launch_info(ordinal, "shallenge");

        let mut rng = rand::thread_rng();
        let stream = &gpu.stream;

        // Allocate static input buffer once (username doesn't change between iterations)
        let username_dev = DeviceBuffer::from_host(stream, username_bytes)?;
        let _ = username_len;

        let launch_config = LaunchConfig {
            grid_dim: (gpu.blocks_per_grid as u32, 1, 1),
            block_dim: (gpu.threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        loop {
            let rng_seed: u64 = rng.r#gen::<u64>();

            // Get the current best hash (with minimal lock time)
            // Use unwrap_or_else to recover data even if lock is poisoned
            let current_target = {
                let best_hash_guard = shared_best_hash.read().unwrap_or_else(|e| e.into_inner());
                best_hash_guard.get_current()
            };

            let mut found_matches_slice = [0u32; 1];
            let mut found_hash = [0u8; 32];
            let mut found_nonce = [0u8; 64];
            let mut found_nonce_len = [0usize; 1];
            let mut found_thread_idx_slice = [0u32; 1];

            // target_hash_dev stays in loop - it changes when a better hash is found
            let target_hash_dev = DeviceBuffer::from_host(stream, &current_target)?;
            let mut found_matches_slice_dev = DeviceBuffer::<u32>::zeroed(stream, 1)?;
            let mut found_hash_dev = DeviceBuffer::<u8>::zeroed(stream, 32)?;
            let mut found_nonce_dev = DeviceBuffer::<u8>::zeroed(stream, 64)?;
            let mut found_nonce_len_dev = DeviceBuffer::<usize>::zeroed(stream, 1)?;
            let mut found_thread_idx_slice_dev = DeviceBuffer::<u32>::zeroed(stream, 1)?;

            unsafe {
                module.kernel_find_better_shallenge_nonce(
                    stream,
                    launch_config,
                    &username_dev,
                    &target_hash_dev,
                    rng_seed,
                    &mut found_matches_slice_dev,
                    &mut found_hash_dev,
                    &mut found_nonce_dev,
                    &mut found_nonce_len_dev,
                    &mut found_thread_idx_slice_dev,
                )?;
            }

            found_matches_slice_dev.copy_to_host(stream, &mut found_matches_slice)?;
            stream.synchronize()?;
            global_stats.add_launch(gpu.operations_per_launch);

            let found_matches = found_matches_slice[0];
            if found_matches != 0 {
                found_hash_dev.copy_to_host(stream, &mut found_hash)?;
                found_nonce_dev.copy_to_host(stream, &mut found_nonce)?;
                found_nonce_len_dev.copy_to_host(stream, &mut found_nonce_len)?;
                found_thread_idx_slice_dev
                    .copy_to_host(stream, &mut found_thread_idx_slice)?;
                stream.synchronize()?;

                let found_thread_idx = found_thread_idx_slice[0];
                let nonce_len = found_nonce_len[0];
                let nonce_string = String::from_utf8(found_nonce[..nonce_len].to_vec()).unwrap();

                // Try to update the global best hash
                let was_global_best = {
                    let mut best_hash_guard = shared_best_hash.write().unwrap_or_else(|e| e.into_inner());
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
