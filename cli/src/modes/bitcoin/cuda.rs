use crate::runner::progress::GlobalStats;
use std::error::Error;
use std::sync::Arc;

use crate::runner::cuda::context::GpuContext;
use cust::launch;
use cust::memory::CopyDestination;
use cust::util::SliceExt;
use rand::Rng;

pub fn run(
    ordinal: usize,
    prefix: String,
    suffix: String,
    gpu: &GpuContext,
    global_stats: Arc<GlobalStats>,
    control: Arc<crate::runner::session::SearchControl>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let prefix_bytes = prefix.as_bytes().to_vec();
    let suffix_bytes = suffix.as_bytes().to_vec();

    let module = gpu.module()?;
    let kernel = module.get_function("kernel_find_bitcoin_vanity_private_key")?;
    gpu.print_launch_info(ordinal, "bitcoin vanity");

    let mut rng = rand::thread_rng();

    // Allocate static input buffers once (they don't change between iterations)
    let prefix_dev = prefix_bytes.as_slice().as_dbuf()?;
    let suffix_dev = suffix_bytes.as_slice().as_dbuf()?;

    while !control.stopped() {
        let rng_seed: u64 = rng.r#gen::<u64>();

        let mut found_matches_slice = [0u32; 1];
        let mut found_private_key = [0u8; 32];
        let mut found_public_key = [0u8; 33];
        let mut found_public_key_hash = [0u8; 20];
        let mut found_encoded_public_key = [0u8; 64];
        let mut found_encoded_len_slice = [0u32; 1];
        let mut found_thread_idx_slice = [0u32; 1];
        let found_matches_dev = found_matches_slice.as_dbuf()?;
        let found_private_key_dev = found_private_key.as_dbuf()?;
        let found_public_key_dev = found_public_key.as_dbuf()?;
        let found_public_key_hash_dev = found_public_key_hash.as_dbuf()?;
        let found_encoded_public_key_dev = found_encoded_public_key.as_dbuf()?;
        let found_encoded_len_dev = found_encoded_len_slice.as_dbuf()?;
        let found_thread_idx_dev = found_thread_idx_slice.as_dbuf()?;

        let stream = &gpu.stream;
        unsafe {
            launch!(
                kernel<<<gpu.blocks_per_grid as u32, gpu.threads_per_block as u32, 0, stream>>>(
                    prefix_dev.as_device_ptr(),
                    prefix_bytes.len(),
                    suffix_dev.as_device_ptr(),
                    suffix_bytes.len(),
                    rng_seed,
                    found_matches_dev.as_device_ptr(),
                    found_private_key_dev.as_device_ptr(),
                    found_public_key_dev.as_device_ptr(),
                    found_public_key_hash_dev.as_device_ptr(),
                    found_encoded_public_key_dev.as_device_ptr(),
                    found_encoded_len_dev.as_device_ptr(),
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
            found_public_key_hash_dev.copy_to(&mut found_public_key_hash)?;
            found_encoded_public_key_dev.copy_to(&mut found_encoded_public_key)?;
            found_encoded_len_dev.copy_to(&mut found_encoded_len_slice)?;
            found_thread_idx_dev.copy_to(&mut found_thread_idx_slice)?;

            // TODO: CPU-verify GPU results before slicing, formatting, or printing:
            // validate encoded_len and candidate metadata, derive the public key,
            // HASH160, and Bech32 address from the private key, compare returned
            // fields, and confirm the requested prefix/suffix match.
            let found_thread_idx = found_thread_idx_slice[0];
            let encoded_len = found_encoded_len_slice[0] as usize;
            let encoded_public_key_str =
                String::from_utf8(found_encoded_public_key[0..encoded_len].to_vec())
                    .unwrap_or_else(|_| "invalid_utf8".to_string());

            // Generate WIF private key format
            let mut encoded_private_key = [0u8; 64];
            let encoded_private_key_len = logic::modes::bitcoin::private_key_to_wif(
                &found_private_key,
                true,
                false,
                &mut encoded_private_key,
            );
            let encoded_private_key_str =
                String::from_utf8(encoded_private_key[0..encoded_private_key_len].to_vec())
                    .unwrap_or_else(|_| "invalid_utf8".to_string());

            println!("[{ordinal}] Vanity match: seed = {rng_seed} thread_idx = {found_thread_idx}");
            println!("[{ordinal}] Vanity match: encoded_public_key = {encoded_public_key_str}");
            println!(
                "[{ordinal}] Vanity match: public_key = {}",
                hex::encode(found_public_key)
            );
            println!(
                "[{ordinal}] Vanity match: public_key_hash = {}",
                hex::encode(found_public_key_hash)
            );
            println!(
                "[{ordinal}] Vanity match: private_key = {}",
                hex::encode(found_private_key)
            );
            println!("[{ordinal}] Vanity match: wallet = {encoded_private_key_str}");

            global_stats.add_matches(found_matches_slice[0] as usize);
        }
    }
    Ok(())
}
