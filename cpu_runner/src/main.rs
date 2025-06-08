use rand::Rng;
use std::error::Error;
use std::sync::{Arc, RwLock};

use common::GlobalStats;

#[derive(Debug, Clone)]
enum Mode {
    SolanaVanity { prefix: String, suffix: String },
    BitcoinVanity { prefix: String, suffix: String },
    Shallenge { username: String, target_hash: String },
}

// Shared state for the best hash found so far
struct SharedBestHash {
    hash: [u8; 32],
}

impl SharedBestHash {
    fn new(initial_hash: [u8; 32]) -> Self {
        Self { hash: initial_hash }
    }
    
    fn update_if_better(&mut self, new_hash: [u8; 32]) -> bool {
        // Compare hashes lexicographically (smaller is better)
        if new_hash < self.hash {
            self.hash = new_hash;
            true
        } else {
            false
        }
    }
    
    fn get_current(&self) -> [u8; 32] {
        self.hash
    }
}

fn validate_base58_string(base58_string: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let invalid_characters = ["l", "I", "0", "O"];
    for invalid_character in invalid_characters {
        if base58_string.contains(invalid_character) {
            return Err(format!("base58 string contains invalid character: {}", invalid_character).into());
        }
    }
    Ok(())
}

fn validate_hex_string(hex_string: &str) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    if hex_string.len() != 64 {
        return Err("Hash must be 64 hex characters (32 bytes)".into());
    }
    hex::decode(hex_string).map_err(|e| format!("Invalid hex string: {}", e).into())
}

fn cpu_worker_thread_solana_vanity(
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
        
        let request = logic::SolanaVanityKeyRequest {
            prefix: vanity_prefix_bytes,
            suffix: vanity_suffix_bytes,
            thread_idx: thread_id,
            rng_seed: rng_seed
        };
        
        let result = logic::generate_and_check_solana_vanity_key(&request);

        global_stats.add_launch(1);
        
        if result.matches {
            let encoded_str = std::str::from_utf8(&result.encoded_public_key[0..result.encoded_len])
                .unwrap_or("invalid_utf8");
            
            println!("[CPU-{}] Vanity match: rng_seed = {}", thread_id, rng_seed);
            println!("[CPU-{}] Vanity match: thread_idx = {}", thread_id, thread_id);
            println!("[CPU-{}] Vanity match: encoded_public_key = {}", thread_id, encoded_str);
            println!("[CPU-{}] Vanity match: public_key = {}", thread_id, hex::encode(result.public_key));
            println!("[CPU-{}] Vanity match: private_key = {}", thread_id, hex::encode(result.private_key));
            println!("[CPU-{}] Vanity match: wallet = {}", thread_id, hex::encode([result.private_key, result.public_key].concat()));

            global_stats.add_matches(1);
            global_stats.print_stats(thread_id, 1);
        }    
    }
}

fn cpu_worker_thread_bitcoin_vanity(
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
        
        let request = logic::BitcoinVanityKeyRequest {
            prefix: vanity_prefix_bytes,
            suffix: vanity_suffix_bytes,
            thread_idx: thread_id,
            rng_seed: rng_seed
        };
        
        let result = logic::generate_and_check_bitcoin_vanity_key(&request);

        global_stats.add_launch(1);
        
        if result.matches {
            let encoded_public_key_str = std::str::from_utf8(&result.encoded_public_key[0..result.encoded_len])
                .unwrap_or("invalid_utf8");
            let mut encoded_private_key = [0u8; 64];
            let encoded_len = logic::private_key_to_wif(&result.private_key, true, false, &mut encoded_private_key);
            let encoded_private_key = &encoded_private_key[0..encoded_len];
            let encoded_private_key_str = std::str::from_utf8(encoded_private_key)
                .unwrap_or("invalid_utf8");
            
            println!("[CPU-{}] Vanity match: rng_seed = {}", thread_id, rng_seed);
            println!("[CPU-{}] Vanity match: thread_idx = {}", thread_id, thread_id);
            println!("[CPU-{}] Vanity match: encoded_public_key = {}", thread_id, encoded_public_key_str);
            println!("[CPU-{}] Vanity match: public_key = {}", thread_id, hex::encode(result.public_key));
            println!("[CPU-{}] Vanity match: public_key_hash = {}", thread_id, hex::encode(result.public_key_hash));
            println!("[CPU-{}] Vanity match: private_key = {}", thread_id, hex::encode(result.private_key));
            println!("[CPU-{}] Vanity match: wallet = {}", thread_id, encoded_private_key_str);

            global_stats.add_matches(1);
            global_stats.print_stats(thread_id, 1);
        }    
    }
}

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

fn cpu_main_solana_vanity(
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
            cpu_worker_thread_solana_vanity(
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

fn cpu_main_bitcoin_vanity(
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
            cpu_worker_thread_bitcoin_vanity(
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

fn cpu_main_shallenge(
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

fn cpu_main(
    num_threads: usize,
    mode: Mode,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    match mode {
        Mode::SolanaVanity { prefix, suffix } => {
            cpu_main_solana_vanity(num_threads, prefix, suffix, global_stats)
        }
        Mode::BitcoinVanity { prefix, suffix } => {
            cpu_main_bitcoin_vanity(num_threads, prefix, suffix, global_stats)
        }
        Mode::Shallenge { username, target_hash } => {
            let target_hash_bytes = validate_hex_string(&target_hash)?;
            cpu_main_shallenge(num_threads, username, target_hash_bytes, global_stats)
        }
    }
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = std::env::args().collect::<Vec<String>>();
    
    let mode = if args.len() == 4 && args[1] == "solana-vanity" {
        let vanity_prefix = args[2].clone();
        let vanity_suffix = args[3].clone();
        validate_base58_string(&vanity_prefix)?;
        validate_base58_string(&vanity_suffix)?;
        Mode::SolanaVanity { prefix: vanity_prefix, suffix: vanity_suffix }
    } else if args.len() == 4 && args[1] == "bitcoin-vanity" {
        let vanity_prefix = args[2].clone();
        let vanity_suffix = args[3].clone();
        validate_base58_string(&vanity_prefix)?;
        validate_base58_string(&vanity_suffix)?;
        Mode::BitcoinVanity { prefix: vanity_prefix, suffix: vanity_suffix }
    } else if args.len() == 4 && args[1] == "shallenge" {
        let username = args[2].clone();
        let target_hash = args[3].clone();
        Mode::Shallenge { username, target_hash }
    } else {
        println!("Usage:");
        println!("  {} solana-vanity <prefix> <suffix>", args[0]);
        println!("  {} bitcoin-vanity <prefix> <suffix>", args[0]);        
        println!("  {} shallenge <username> <target_hash_hex>", args[0]);
        std::process::exit(1);
    };

    let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

    let global_stats = match &mode {
        Mode::SolanaVanity { prefix, suffix } => Arc::new(GlobalStats::new(
            num_threads,
            prefix.len(),
            suffix.len()
        )),
        Mode::BitcoinVanity { prefix, suffix } => Arc::new(GlobalStats::new(
            num_threads,
            prefix.len(),
            suffix.len()
        )),
        Mode::Shallenge { username, .. } => Arc::new(GlobalStats::new(
            num_threads,
            username.len(),
            0 // No suffix for shallenge
        )),
    };

    match &mode {
        Mode::SolanaVanity { prefix, suffix } => {
            println!("Searching for solana vanity key with prefix '{}' and suffix '{}'", prefix, suffix);
        }
        Mode::BitcoinVanity { prefix, suffix } => {
            println!("Searching for bitcoin vanity key with prefix '{}' and suffix '{}'", prefix, suffix);
        }
        Mode::Shallenge { username, target_hash } => {
            println!("Starting shallenge for username '{}' with target hash '{}'", username, target_hash);
        }
    }

    // Launch appropriate compute mode
    cpu_main(num_threads, mode, global_stats)
}
