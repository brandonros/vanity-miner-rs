use rand::Rng;
use std::error::Error;
use std::sync::Arc;

use common::GlobalStats;

fn validate_base58_string(base58_string: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let invalid_characters = ["l", "I", "0", "O"];
    for invalid_character in invalid_characters {
        if base58_string.contains(invalid_character) {
            return Err(format!("base58 string contains invalid character: {}", invalid_character).into());
        }
    }
    Ok(())
}

fn cpu_worker_thread(
    thread_id: usize,
    vanity_prefix: String,
    vanity_suffix: String,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut rng = rand::thread_rng();
    let vanity_prefix_bytes = vanity_prefix.as_bytes();
    let vanity_suffix_bytes = vanity_suffix.as_bytes();
    
    println!("[CPU-{}] Starting CPU worker thread", thread_id);
    
    loop {
        let rng_seed: u64 = rng.r#gen();
        
        let request = logic::VanityKeyRequest {
            prefix: vanity_prefix_bytes,
            suffix: vanity_suffix_bytes,
            thread_idx: thread_id,
            rng_seed: rng_seed
        };
        
        let result = logic::generate_and_check_vanity_key(&request);

        global_stats.add_launch(1);
        
        if result.matches {
            let encoded_str = std::str::from_utf8(&result.encoded_public_key[0..result.encoded_len])
                .unwrap_or("invalid_utf8");
            
            println!("[CPU-{}] First match: thread_idx = {}", thread_id, thread_id);
            println!("[CPU-{}] First match: encoded_public_key = {}", thread_id, encoded_str);
            println!("[CPU-{}] First match: public_key = {}", thread_id, hex::encode(result.public_key));
            println!("[CPU-{}] First match: hashed_private_key = {}", thread_id, hex::encode(result.hashed_private_key));                    
            println!("[CPU-{}] First match: private_key = {}", thread_id, hex::encode(result.private_key));
            println!("[CPU-{}] First match: wallet = {}", thread_id, hex::encode([result.private_key, result.public_key].concat()));

            global_stats.add_matches(1);
            global_stats.print_stats(thread_id, 1.0);
        }    
    }

    Ok(())
}

fn cpu_main(
    num_threads: usize,
    vanity_prefix: String,
    vanity_suffix: String,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("Starting CPU mode with {} threads", num_threads);
    
    let mut handles = Vec::new();
    
    // Start CPU worker threads
    for i in 0..num_threads {
        let vanity_prefix_clone = vanity_prefix.clone();
        let vanity_suffix_clone = vanity_suffix.clone();
        let stats_clone = Arc::clone(&global_stats);
        
        handles.push(std::thread::spawn(move || {
            cpu_worker_thread(
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

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() < 3 || args.len() > 4 {
        println!("Usage: {} <vanity_prefix> <vanity_suffix>", args[0]);
        std::process::exit(1);
    }
    
    let vanity_prefix = args[1].to_string();
    let vanity_suffix = args[2].to_string();
    validate_base58_string(&vanity_prefix)?;
    validate_base58_string(&vanity_suffix)?;

    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    let global_stats = Arc::new(GlobalStats::new(
        num_threads,
        vanity_prefix.len(),
        vanity_suffix.len(),
    ));

    println!("Searching for vanity key with prefix '{}' and suffix '{}'", vanity_prefix, vanity_suffix);

    // Launch appropriate compute mode
    cpu_main(num_threads, vanity_prefix, vanity_suffix, global_stats)
}
