use std::error::Error;
use std::sync::Arc;

/// Spawns `num_workers` threads, each running `worker_fn(thread_id, shared_data)`.
/// Waits for all to complete. Propagates errors without panicking.
pub fn spawn_cpu_workers<T, F>(
    num_workers: usize,
    shared_data: Arc<T>,
    worker_fn: F,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    T: Send + Sync + 'static,
    F: Fn(usize, Arc<T>) -> Result<(), Box<dyn Error + Send + Sync>> + Send + Clone + 'static,
{
    let handles: Vec<_> = (0..num_workers)
        .map(|i| {
            let data = Arc::clone(&shared_data);
            let f = worker_fn.clone();
            std::thread::spawn(move || f(i, data))
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .map_err(|_| format!("Worker thread {} panicked", i))?
            .map_err(|e| format!("Worker thread {} failed: {}", i, e))?;
    }

    Ok(())
}
