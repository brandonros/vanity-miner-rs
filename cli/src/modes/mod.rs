use crate::common::GlobalStats;
use rand::Rng as _;
use std::error::Error;
use std::sync::Arc;

pub mod bitcoin;
pub mod ethereum;
pub mod shallenge;
pub mod solana;

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
