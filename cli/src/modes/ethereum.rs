use crate::common::GlobalStats;
use crate::modes::VanityMode;
use std::error::Error;
use std::sync::Arc;

/// Ethereum vanity address mode
pub struct EthereumVanity;

impl VanityMode for EthereumVanity {
    type Result = logic::EthereumVanityKeyResult;

    const NAME: &'static str = "ethereum vanity";

    fn process_prefix(prefix: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
        Ok(hex::decode(prefix)?)
    }

    fn process_suffix(suffix: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
        Ok(hex::decode(suffix)?)
    }

    fn generate_and_check(
        prefix: &[u8],
        suffix: &[u8],
        thread_idx: usize,
        rng_seed: u64,
    ) -> Self::Result {
        let request = logic::EthereumVanityKeyRequest {
            prefix,
            suffix,
            thread_idx,
            rng_seed,
        };
        logic::generate_and_check_ethereum_vanity_key(&request)
    }

    fn is_match(result: &Self::Result) -> bool {
        result.matches
    }

    fn print_match(thread_id: usize, rng_seed: u64, result: &Self::Result) {
        let encoded_address_str = hex::encode(result.address);

        println!("[CPU-{thread_id}] Vanity match: rng_seed = {rng_seed}");
        println!("[CPU-{thread_id}] Vanity match: thread_idx = {thread_id}");
        println!("[CPU-{thread_id}] Vanity match: address = 0x{encoded_address_str}");
        println!("[CPU-{thread_id}] Vanity match: public_key = {}", hex::encode(result.public_key));
        println!("[CPU-{thread_id}] Vanity match: private_key = 0x{}", hex::encode(result.private_key));
        println!("[CPU-{thread_id}] Vanity match: wallet = 0x{}", hex::encode(result.private_key));
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
        crate::modes::run_cpu_vanity::<EthereumVanity>(num_threads, vanity_prefix, vanity_suffix, global_stats)
    }
}

#[cfg(feature = "gpu")]
pub use gpu_impl::*;

#[cfg(feature = "gpu")]
mod gpu_impl {
    use super::*;
    use crate::common::GpuContext;
    use crate::modes::{GpuBuffers, GpuVanityMode};
    use cust::function::Function;
    use cust::launch;
    use cust::memory::CopyDestination;
    use cust::util::SliceExt;

    /// GPU buffer layout for Ethereum vanity mining
    #[derive(Default)]
    pub struct EthereumGpuBuffers {
        pub found_matches_slice: [u32; 1],
        pub found_private_key: [u8; 32],
        pub found_public_key: [u8; 64],
        pub found_address: [u8; 20],
        pub found_thread_idx_slice: [u32; 1],
    }

    impl GpuBuffers for EthereumGpuBuffers {
        fn run_iteration(
            &mut self,
            kernel: &Function,
            gpu: &GpuContext,
            prefix_bytes: &[u8],
            suffix_bytes: &[u8],
            rng_seed: u64,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            let prefix_len = prefix_bytes.len();
            let suffix_len = suffix_bytes.len();

            let prefix_dev = prefix_bytes.as_dbuf()?;
            let suffix_dev = suffix_bytes.as_dbuf()?;
            let found_matches_dev = self.found_matches_slice.as_dbuf()?;
            let found_private_key_dev = self.found_private_key.as_dbuf()?;
            let found_public_key_dev = self.found_public_key.as_dbuf()?;
            let found_address_dev = self.found_address.as_dbuf()?;
            let found_thread_idx_dev = self.found_thread_idx_slice.as_dbuf()?;

            unsafe {
                launch!(
                    kernel<<<gpu.blocks_per_grid as u32, gpu.threads_per_block as u32, 0, gpu.stream>>>(
                        prefix_dev.as_device_ptr(),
                        prefix_len,
                        suffix_dev.as_device_ptr(),
                        suffix_len,
                        rng_seed,
                        found_matches_dev.as_device_ptr(),
                        found_private_key_dev.as_device_ptr(),
                        found_public_key_dev.as_device_ptr(),
                        found_address_dev.as_device_ptr(),
                        found_thread_idx_dev.as_device_ptr(),
                    )
                )?;
            }

            gpu.stream.synchronize()?;

            // Always copy back matches count
            found_matches_dev.copy_to(&mut self.found_matches_slice)?;

            // Only copy remaining data if we found a match
            if self.found_matches_slice[0] != 0 {
                found_private_key_dev.copy_to(&mut self.found_private_key)?;
                found_public_key_dev.copy_to(&mut self.found_public_key)?;
                found_address_dev.copy_to(&mut self.found_address)?;
                found_thread_idx_dev.copy_to(&mut self.found_thread_idx_slice)?;
            }

            Ok(())
        }

        fn found_matches(&self) -> u32 {
            self.found_matches_slice[0]
        }

        fn found_thread_idx(&self) -> u32 {
            self.found_thread_idx_slice[0]
        }
    }

    impl GpuVanityMode for EthereumVanity {
        const KERNEL_NAME: &'static str = "kernel_find_ethereum_vanity_private_key";
        const NAME: &'static str = "ethereum vanity";
        type Buffers = EthereumGpuBuffers;

        fn process_prefix(prefix: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
            Ok(hex::decode(prefix)?)
        }

        fn process_suffix(suffix: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
            Ok(hex::decode(suffix)?)
        }

        fn print_match(ordinal: usize, rng_seed: u64, buffers: &Self::Buffers) {
            let found_thread_idx = buffers.found_thread_idx();
            let encoded_address_str = hex::encode(buffers.found_address);

            println!("[{ordinal}] Vanity match: seed = {rng_seed} thread_idx = {found_thread_idx}");
            println!("[{ordinal}] Vanity match: address = 0x{encoded_address_str}");
            println!("[{ordinal}] Vanity match: public_key = {}", hex::encode(buffers.found_public_key));
            println!("[{ordinal}] Vanity match: private_key = 0x{}", hex::encode(buffers.found_private_key));
            println!("[{ordinal}] Vanity match: wallet = 0x{}", hex::encode(buffers.found_private_key));
        }
    }

    pub fn run(
        ordinal: usize,
        prefix: String,
        suffix: String,
        module: &cust::module::Module,
        global_stats: Arc<GlobalStats>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        crate::modes::run_gpu_vanity::<EthereumVanity>(ordinal, prefix, suffix, module, global_stats)
    }
}
