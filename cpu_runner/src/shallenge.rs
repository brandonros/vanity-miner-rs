use std::error::Error;
use std::sync::{Arc, RwLock};

use common::GlobalStats;
use rand::Rng as _;

use crate::shared_best_hash::SharedBestHash;

fn cpu_worker_thread_shallenge(
    thread_id: usize,
    username: String,
    shared_best_hash: Arc<RwLock<SharedBestHash>>,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut rng = rand::thread_rng();
    
    println!("[CPU-{}] Starting CPU shallenge worker thread", thread_id);
    
    loop {
        let rng_seed: u64 = rng.r#gen();
        
        // Get the current best hash (with minimal lock time)
        let current_target = {
            let best_hash_guard = shared_best_hash.read().unwrap();
            best_hash_guard.get_current()
        };
        
        // Create the request with the current best target
        let request = logic::ShallengeRequest {
            username: username.as_bytes(),
            username_len: username.len(),
            target_hash: &current_target,
            thread_idx: thread_id,
            rng_seed: rng_seed
        };
        
        let result = logic::generate_and_check_shallenge(&request);

        global_stats.add_launch(1);
        
        if result.is_better {
            let nonce_string = std::str::from_utf8(&result.nonce[0..result.nonce_len])
                .unwrap_or("invalid_utf8");
            
            // Try to update the global best hash
            let was_global_best = {
                let mut best_hash_guard = shared_best_hash.write().unwrap();
                best_hash_guard.update_if_better(result.hash)
            };
            
            if was_global_best {
                println!("[CPU-{}] NEW GLOBAL BEST found: thread_idx = {}", thread_id, thread_id);
                println!("[CPU-{}] NEW GLOBAL BEST hash: {}", thread_id, hex::encode(result.hash));
                println!("[CPU-{}] NEW GLOBAL BEST nonce: {}", thread_id, nonce_string);
                println!("[CPU-{}] Challenge string: {}/{}", thread_id, username, nonce_string);

                global_stats.add_matches(1);
                global_stats.print_stats(thread_id, 1);
            }
            // If it wasn't a global best, another thread found something better in the meantime
            // Just continue without printing - this prevents spam from outdated results
        }    
    }
}

pub fn cpu_main_shallenge(
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
    
    let mut handles = Vec::new();
    
    // Start CPU worker threads
    for i in 0..num_threads {
        let username_clone = username.clone();
        let shared_best_hash_clone = Arc::clone(&shared_best_hash);
        let stats_clone = Arc::clone(&global_stats);
        
        handles.push(std::thread::spawn(move || {
            cpu_worker_thread_shallenge(
                i,
                username_clone,
                shared_best_hash_clone,
                stats_clone,
            )
        }));
    }
    
    // Wait for all threads (in practice, they run forever)
    for handle in handles {
        handle.join().unwrap()?;
    }
    
    Ok(())
}
