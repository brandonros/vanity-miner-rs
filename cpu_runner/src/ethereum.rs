use std::error::Error;
use std::sync::Arc;
use rand::Rng as _;

use common::GlobalStats;

fn cpu_worker_thread_ethereum_vanity(
    thread_id: usize,
    vanity_prefix: String,
    vanity_suffix: String,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut rng = rand::thread_rng();
    let vanity_prefix_bytes = vanity_prefix.as_bytes();
    let vanity_suffix_bytes = vanity_suffix.as_bytes();
    
    println!("[CPU-{}] Starting CPU vanity worker thread", thread_id);
    
    loop {
        let rng_seed: u64 = rng.r#gen();
        
        let request = logic::EthereumVanityKeyRequest {
            prefix: vanity_prefix_bytes,
            suffix: vanity_suffix_bytes,
            thread_idx: thread_id,
            rng_seed: rng_seed
        };
        
        let result = logic::generate_and_check_ethereum_vanity_key(&request);

        global_stats.add_launch(1);
        
        if result.matches {
            let encoded_address_str = std::str::from_utf8(&result.encoded_address)
                .unwrap_or("invalid_utf8");
            
            println!("[CPU-{}] Vanity match: rng_seed = {}", thread_id, rng_seed);
            println!("[CPU-{}] Vanity match: thread_idx = {}", thread_id, thread_id);
            println!("[CPU-{}] Vanity match: address = {}", thread_id, encoded_address_str);
            println!("[CPU-{}] Vanity match: public_key = {}", thread_id, hex::encode(result.public_key));
            println!("[CPU-{}] Vanity match: private_key = 0x{}", thread_id, hex::encode(result.private_key));
            println!("[CPU-{}] Vanity match: wallet = 0x{}", thread_id, hex::encode(result.private_key)); // Same as private key
            
            global_stats.add_matches(1);
            global_stats.print_stats(thread_id, 1);
        }
    }
}

pub fn cpu_main_ethereum_vanity(
    num_threads: usize,
    vanity_prefix: String,
    vanity_suffix: String,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("Starting CPU vanity mode with {} threads", num_threads);
    
    let mut handles = Vec::new();
    
    // Start CPU worker threads
    for i in 0..num_threads {
        let vanity_prefix_clone = vanity_prefix.clone();
        let vanity_suffix_clone = vanity_suffix.clone();
        let stats_clone = Arc::clone(&global_stats);
        
        handles.push(std::thread::spawn(move || {
            cpu_worker_thread_ethereum_vanity(
                i,
                vanity_prefix_clone,
                vanity_suffix_clone,
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
