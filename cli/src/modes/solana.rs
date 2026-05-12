use crate::common::GlobalStats;
use std::error::Error;
use std::sync::Arc;

#[cfg(not(feature = "gpu"))]
pub mod cpu {
    use super::*;
    use crate::common::spawn_cpu_workers;
    use rand::Rng as _;

    struct WorkerData {
        prefix_bytes: Vec<u8>,
        suffix_bytes: Vec<u8>,
        global_stats: Arc<GlobalStats>,
    }

    fn worker(thread_id: usize, data: Arc<WorkerData>) -> Result<(), Box<dyn Error + Send + Sync>> {
        let mut rng = rand::thread_rng();

        println!("[CPU-{thread_id}] Starting CPU solana vanity worker thread");

        loop {
            let rng_seed: u64 = rng.r#gen();

            let request = logic::SolanaVanityKeyRequest {
                prefix: &data.prefix_bytes,
                suffix: &data.suffix_bytes,
                thread_idx: thread_id,
                rng_seed,
            };

            let result = logic::generate_and_check_solana_vanity_key(&request);

            data.global_stats.add_launch(1);

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

                data.global_stats.add_matches(1);
                data.global_stats.print_stats(thread_id, 1);
            }
        }
    }

    pub fn run(
        num_threads: usize,
        prefix: String,
        suffix: String,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        println!("Starting CPU solana vanity mode with {} threads", num_threads);

        let data = Arc::new(WorkerData {
            prefix_bytes: prefix.as_bytes().to_vec(),
            suffix_bytes: suffix.as_bytes().to_vec(),
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
        prefix: String,
        suffix: String,
        ctx: &Arc<CudaContext>,
        module: &LoadedModule,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let prefix_bytes = prefix.as_bytes().to_vec();
        let suffix_bytes = suffix.as_bytes().to_vec();

        let gpu = GpuContext::new(ctx)?;
        gpu.print_launch_info(ordinal, "solana vanity");

        let mut rng = rand::thread_rng();
        let stream = &gpu.stream;

        // Allocate static input buffers once (they don't change between iterations)
        let prefix_dev = DeviceBuffer::from_host(stream, prefix_bytes.as_slice())?;
        let suffix_dev = DeviceBuffer::from_host(stream, suffix_bytes.as_slice())?;

        let launch_config = LaunchConfig {
            grid_dim: (gpu.blocks_per_grid as u32, 1, 1),
            block_dim: (gpu.threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        loop {
            let rng_seed: u64 = rng.r#gen::<u64>();

            let mut found_matches_slice = [0u32; 1];
            let mut found_private_key = [0u8; 32];
            let mut found_public_key = [0u8; 32];
            let mut found_encoded_public_key = [0u8; 64];
            let mut found_thread_idx_slice = [0u32; 1];
            let mut found_matches_dev = DeviceBuffer::<u32>::zeroed(stream, 1)?;
            let mut found_private_key_dev = DeviceBuffer::<u8>::zeroed(stream, 32)?;
            let mut found_public_key_dev = DeviceBuffer::<u8>::zeroed(stream, 32)?;
            let mut found_encoded_public_key_dev = DeviceBuffer::<u8>::zeroed(stream, 64)?;
            let mut found_thread_idx_dev = DeviceBuffer::<u32>::zeroed(stream, 1)?;

            unsafe {
                module.kernel_find_solana_vanity_private_key(
                    stream,
                    launch_config,
                    &prefix_dev,
                    &suffix_dev,
                    rng_seed,
                    &mut found_matches_dev,
                    &mut found_private_key_dev,
                    &mut found_public_key_dev,
                    &mut found_encoded_public_key_dev,
                    &mut found_thread_idx_dev,
                )?;
            }

            found_matches_dev.copy_to_host(stream, &mut found_matches_slice)?;
            stream.synchronize()?;
            global_stats.add_launch(gpu.operations_per_launch);

            if found_matches_slice[0] != 0 {
                found_private_key_dev.copy_to_host(stream, &mut found_private_key)?;
                found_public_key_dev.copy_to_host(stream, &mut found_public_key)?;
                found_encoded_public_key_dev
                    .copy_to_host(stream, &mut found_encoded_public_key)?;
                found_thread_idx_dev.copy_to_host(stream, &mut found_thread_idx_slice)?;
                stream.synchronize()?;

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
