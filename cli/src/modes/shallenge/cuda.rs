use crate::common::GlobalStats;
use crate::modes::shallenge::shared_best_hash::SharedBestHash;
use std::error::Error;
use std::sync::{Arc, RwLock};

use crate::common::GpuContext;
use cust::launch;
use cust::memory::CopyDestination;
use cust::util::SliceExt;
use rand::Rng;

pub fn run(
    ordinal: usize,
    username: String,
    shared_best_hash: Arc<RwLock<SharedBestHash>>,
    gpu: &GpuContext,
    global_stats: Arc<GlobalStats>,
    control: Arc<vanity_miner::search_control::SearchControl>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let username_bytes = username.as_bytes();
    let username_len: usize = username_bytes.len();

    let module = &gpu.module;
    let kernel = module.get_function("kernel_find_better_shallenge_nonce")?;
    gpu.print_launch_info(ordinal, "shallenge");

    let mut rng = rand::thread_rng();

    // Allocate static input buffer once (username doesn't change between iterations)
    let username_dev = username_bytes.as_dbuf()?;

    while !control.stopped() {
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
        let target_hash_dev = current_target.as_dbuf()?;
        let found_matches_slice_dev = found_matches_slice.as_dbuf()?;
        let found_hash_dev = found_hash.as_dbuf()?;
        let found_nonce_dev = found_nonce.as_dbuf()?;
        let found_nonce_len_dev = found_nonce_len.as_dbuf()?;
        let found_thread_idx_slice_dev = found_thread_idx_slice.as_dbuf()?;

        let stream = &gpu.stream;
        unsafe {
            launch!(
                kernel<<<gpu.blocks_per_grid as u32, gpu.threads_per_block as u32, 0, stream>>>(
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

            // TODO: CPU-verify GPU results before slicing, printing, or updating
            // the global target: validate nonce_len and candidate metadata,
            // recompute SHA256(username || '/' || nonce), compare found_hash,
            // and confirm it beats the launch target.
            let found_thread_idx = found_thread_idx_slice[0];
            let nonce_len = found_nonce_len[0];
            let nonce_string = String::from_utf8(found_nonce[..nonce_len].to_vec()).unwrap();

            // Try to update the global best hash
            let was_global_best = {
                let mut best_hash_guard =
                    shared_best_hash.write().unwrap_or_else(|e| e.into_inner());
                best_hash_guard.update_if_better(found_hash)
            };

            if was_global_best {
                println!(
                    "[{ordinal}] NEW GLOBAL BEST found: seed = {rng_seed} thread_idx = {found_thread_idx}"
                );
                println!(
                    "[{ordinal}] NEW GLOBAL BEST hash: {}",
                    hex::encode(found_hash)
                );
                println!("[{ordinal}] NEW GLOBAL BEST nonce: {}", nonce_string);
                println!(
                    "[{ordinal}] Challenge string: {}/{}",
                    username, nonce_string
                );

                global_stats.add_matches(found_matches as usize);
            }
        }
    }
    Ok(())
}
