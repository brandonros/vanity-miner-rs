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
        let vanity_prefix_bytes = hex::decode(vanity_prefix)?;
        let vanity_suffix_bytes = hex::decode(vanity_suffix)?;

        println!("[CPU-{}] Starting CPU vanity worker thread", thread_id);

        loop {
            let rng_seed: u64 = rng.r#gen();

            let request = logic::EthereumVanityKeyRequest {
                prefix: &vanity_prefix_bytes,
                suffix: &vanity_suffix_bytes,
                thread_idx: thread_id,
                rng_seed,
            };

            let result = logic::generate_and_check_ethereum_vanity_key(&request);

            global_stats.add_launch(1);

            if result.matches {
                let encoded_address_str = hex::encode(result.address);

                println!("[CPU-{}] Vanity match: rng_seed = {}", thread_id, rng_seed);
                println!("[CPU-{}] Vanity match: thread_idx = {}", thread_id, thread_id);
                println!("[CPU-{}] Vanity match: address = 0x{}", thread_id, encoded_address_str);
                println!("[CPU-{}] Vanity match: public_key = {}", thread_id, hex::encode(result.public_key));
                println!("[CPU-{}] Vanity match: private_key = 0x{}", thread_id, hex::encode(result.private_key));
                println!("[CPU-{}] Vanity match: wallet = 0x{}", thread_id, hex::encode(result.private_key));

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
        crate::modes::spawn_cpu_workers(num_threads, "vanity", move |thread_id| {
            worker(thread_id, vanity_prefix.clone(), vanity_suffix.clone(), Arc::clone(&global_stats))
        })
    }
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;
    use cust::context::ResourceLimit;
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
        let vanity_prefix_vec = hex::decode(vanity_prefix)?;
        let vanity_prefix_bytes = vanity_prefix_vec.as_slice();
        let vanity_prefix_len: usize = vanity_prefix_bytes.len();
        let vanity_suffix_vec = hex::decode(vanity_suffix)?;
        let vanity_suffix_bytes = vanity_suffix_vec.as_slice();
        let vanity_suffix_len: usize = vanity_suffix_bytes.len();

        let device = Device::get_device(ordinal as u32)?;
        let ctx = Context::new(device)?;
        cust::context::CurrentContext::set_current(&ctx)?;

        // Optionally override stack size
        if let Some(stack_size) = std::env::var("STACK_SIZE").ok() {
            let stack_size = stack_size.parse::<usize>().unwrap();
            cust::context::CurrentContext::set_resource_limit(ResourceLimit::StackSize, stack_size)?;
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let find_ethereum_vanity_private_key =
            module.get_function("kernel_find_ethereum_vanity_private_key")?;

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
            let mut found_public_key = [0u8; 64]; // Ethereum uncompressed public key is 64 bytes
            let mut found_address = [0u8; 20]; // Ethereum address is 20 bytes
            let mut found_thread_idx_slice = [0u32; 1];

            let vanity_prefix_dev = vanity_prefix_bytes.as_dbuf()?;
            let vanity_suffix_dev = vanity_suffix_bytes.as_dbuf()?;
            let found_matches_slice_dev = found_matches_slice.as_dbuf()?;
            let found_private_key_dev = found_private_key.as_dbuf()?;
            let found_public_key_dev = found_public_key.as_dbuf()?;
            let found_address_dev = found_address.as_dbuf()?;
            let found_thread_idx_slice_dev = found_thread_idx_slice.as_dbuf()?;

            unsafe {
                launch!(
                    find_ethereum_vanity_private_key<<<blocks_per_grid as u32, threads_per_block as u32, 0, stream>>>(
                        vanity_prefix_dev.as_device_ptr(),
                        vanity_prefix_len,
                        vanity_suffix_dev.as_device_ptr(),
                        vanity_suffix_len,
                        rng_seed,
                        found_matches_slice_dev.as_device_ptr(),
                        found_private_key_dev.as_device_ptr(),
                        found_public_key_dev.as_device_ptr(),
                        found_address_dev.as_device_ptr(),
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
                found_address_dev.copy_to(&mut found_address)?;
                found_thread_idx_slice_dev.copy_to(&mut found_thread_idx_slice)?;

                let found_thread_idx = found_thread_idx_slice[0];
                let encoded_address_str = hex::encode(found_address);

                println!("[{ordinal}] Vanity match: seed = {rng_seed} thread_idx = {found_thread_idx}");
                println!("[{ordinal}] Vanity match: address = 0x{encoded_address_str}");
                println!("[{ordinal}] Vanity match: public_key = {}", hex::encode(found_public_key));
                println!("[{ordinal}] Vanity match: private_key = 0x{}", hex::encode(found_private_key));
                println!("[{ordinal}] Vanity match: wallet = 0x{}", hex::encode(found_private_key));

                global_stats.add_matches(found_matches as usize);
                global_stats.print_stats(ordinal, found_matches as u32);
            }
        }
    }
}
