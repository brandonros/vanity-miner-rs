use crate::common::GlobalStats;
use rand::Rng as _;
use std::error::Error;
use std::sync::Arc;

pub mod bitcoin;
pub mod ethereum;
pub mod shallenge;
pub mod solana;

#[cfg(feature = "gpu")]
use crate::common::GpuContext;
#[cfg(feature = "gpu")]
use cust::function::Function;
#[cfg(feature = "gpu")]
use cust::module::Module;

/// Trait for vanity address generation modes.
/// Implement this trait to enable generic CPU worker support.
pub trait VanityMode: Send + Sync + 'static {
    type Result: Send;

    /// Human-readable name for the mode (used in logs)
    const NAME: &'static str;

    /// Process the prefix string into bytes.
    /// Default: convert directly to bytes. Override for hex decoding etc.
    fn process_prefix(prefix: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
        Ok(prefix.as_bytes().to_vec())
    }

    /// Process the suffix string into bytes.
    /// Default: convert directly to bytes. Override for hex decoding etc.
    fn process_suffix(suffix: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
        Ok(suffix.as_bytes().to_vec())
    }

    /// Generate a key and check if it matches the vanity pattern.
    fn generate_and_check(
        prefix: &[u8],
        suffix: &[u8],
        thread_idx: usize,
        rng_seed: u64,
    ) -> Self::Result;

    /// Check if the result is a match.
    fn is_match(result: &Self::Result) -> bool;

    /// Print match details to stdout.
    fn print_match(thread_id: usize, rng_seed: u64, result: &Self::Result);
}

/// Generic CPU runner for any VanityMode.
pub fn run_cpu_vanity<M: VanityMode>(
    num_threads: usize,
    prefix: String,
    suffix: String,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Process inputs once before spawning threads
    let prefix_bytes = M::process_prefix(&prefix)?;
    let suffix_bytes = M::process_suffix(&suffix)?;

    spawn_cpu_workers(num_threads, M::NAME, move |thread_id| {
        let mut rng = rand::thread_rng();
        let prefix = prefix_bytes.clone();
        let suffix = suffix_bytes.clone();
        let stats = Arc::clone(&global_stats);

        println!("[CPU-{thread_id}] Starting CPU {} worker thread", M::NAME);

        loop {
            let rng_seed: u64 = rng.r#gen();
            let result = M::generate_and_check(&prefix, &suffix, thread_id, rng_seed);
            stats.add_launch(1);

            if M::is_match(&result) {
                M::print_match(thread_id, rng_seed, &result);
                stats.add_matches(1);
                stats.print_stats(thread_id, 1);
            }
        }
    })
}

/// Spawns CPU worker threads that run the given worker function.
/// Each thread receives its thread_id (0..num_threads).
pub fn spawn_cpu_workers<F>(
    num_threads: usize,
    mode_name: &str,
    worker_fn: F,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    F: Fn(usize) -> Result<(), Box<dyn Error + Send + Sync>> + Send + Sync + Clone + 'static,
{
    println!("Starting CPU {} mode with {} threads", mode_name, num_threads);

    let mut handles = Vec::new();
    for i in 0..num_threads {
        let worker = worker_fn.clone();
        handles.push(std::thread::spawn(move || worker(i)));
    }

    for handle in handles {
        handle.join().unwrap()?;
    }

    Ok(())
}

// ============================================================================
// GPU Traits and Generic Runner
// ============================================================================

/// Trait for GPU vanity address generation modes.
/// Implement this trait to enable generic GPU worker support.
#[cfg(feature = "gpu")]
pub trait GpuVanityMode: Send + Sync + 'static {
    /// Kernel function name in the PTX module
    const KERNEL_NAME: &'static str;

    /// Human-readable name for logging
    const NAME: &'static str;

    /// The buffer type for this mode
    type Buffers: GpuBuffers;

    /// Process prefix string (ASCII bytes vs hex decode)
    fn process_prefix(prefix: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>>;

    /// Process suffix string
    fn process_suffix(suffix: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>>;

    /// Print match result from GPU buffers
    fn print_match(ordinal: usize, rng_seed: u64, buffers: &Self::Buffers);
}

/// Trait for GPU buffer management.
/// Each chain implements this to handle its specific buffer layout and kernel launch.
#[cfg(feature = "gpu")]
pub trait GpuBuffers: Default {
    /// Run one iteration: allocate device buffers, launch kernel, sync, copy back.
    fn run_iteration(
        &mut self,
        kernel: &Function,
        gpu: &GpuContext,
        prefix_bytes: &[u8],
        suffix_bytes: &[u8],
        rng_seed: u64,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// Get found_matches count after copy-back
    fn found_matches(&self) -> u32;

    /// Get thread_idx of the match after copy-back
    fn found_thread_idx(&self) -> u32;
}

/// Generic GPU runner for any GpuVanityMode.
#[cfg(feature = "gpu")]
pub fn run_gpu_vanity<M: GpuVanityMode>(
    ordinal: usize,
    prefix: String,
    suffix: String,
    module: &Module,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let prefix_bytes = M::process_prefix(&prefix)?;
    let suffix_bytes = M::process_suffix(&suffix)?;

    let gpu = GpuContext::new(ordinal)?;
    let kernel = module.get_function(M::KERNEL_NAME)?;
    gpu.print_launch_info(ordinal, "vanity");

    let mut rng = rand::thread_rng();

    loop {
        let rng_seed: u64 = rng.r#gen();

        let mut buffers = M::Buffers::default();
        buffers.run_iteration(&kernel, &gpu, &prefix_bytes, &suffix_bytes, rng_seed)?;

        global_stats.add_launch(gpu.operations_per_launch);

        if buffers.found_matches() != 0 {
            M::print_match(ordinal, rng_seed, &buffers);
            global_stats.add_matches(buffers.found_matches() as usize);
            global_stats.print_stats(ordinal, buffers.found_matches());
        }
    }
}
