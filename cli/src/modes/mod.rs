use std::error::Error;

pub mod solana;
pub mod bitcoin;
pub mod ethereum;
pub mod shallenge;

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
