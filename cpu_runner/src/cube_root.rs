use std::error::Error;
use std::sync::Arc;
use rand::Rng as _;
use common::GlobalStats;

fn cpu_worker_thread_cube_root(
    thread_id: usize,
    message: Vec<u8>,
    modulus: Vec<u8>,
    exponent: Vec<u8>,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut rng = rand::thread_rng();
    
    println!("[CPU-{}] Starting CPU cuberoot worker thread", thread_id);
    
    loop {
        let rng_seed: u64 = rng.r#gen();
        
        // Create the request with the cube root parameters
        let request = logic::CubeRootRequest {
            message: &message,
            message_len: message.len(),
            modulus: &modulus,
            exponent: &exponent,
            thread_idx: thread_id,
            rng_seed: rng_seed
        };
        
        let result = logic::generate_and_check_cuberoot(&request);

        global_stats.add_launch(1);
        
        if result.found_perfect_cube {
            let nonce_string = std::str::from_utf8(&result.nonce[0..result.nonce_len])
                .unwrap_or("invalid_utf8");
            
            println!("[CPU-{}] PERFECT CUBE FOUND!", thread_id);
            println!("[CPU-{}] Message: {}", thread_id, String::from_utf8_lossy(&message));
            println!("[CPU-{}] Nonce: {}", thread_id, nonce_string);
            println!("[CPU-{}] Hash (perfect cube): {}", thread_id, hex::encode(result.hash));
            println!("[CPU-{}] Forged signature: {}", thread_id, hex::encode(result.signature));
            println!("[CPU-{}] Full message: {}/{}", thread_id, 
                String::from_utf8_lossy(&message), nonce_string);

            global_stats.add_matches(1);
            global_stats.print_stats(thread_id, 1);
            
            // Exit successfully - we found what we were looking for!
            std::process::exit(0);
        }    
    }
}

pub fn cpu_main_cube_root(
    num_threads: usize,
    message_bytes: Vec<u8>,
    modulus_bytes: Vec<u8>,
    exponent_bytes: Vec<u8>,        
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("Starting CPU cube root mode with {} threads", num_threads);
    
    let mut handles = Vec::new();
    
    // Start CPU worker threads
    for i in 0..num_threads {
        let message_clone = message_bytes.clone();
        let modulus_clone = modulus_bytes.clone();
        let exponent_clone = exponent_bytes.clone();
        let stats_clone = Arc::clone(&global_stats);
        
        handles.push(std::thread::spawn(move || {
            cpu_worker_thread_cube_root(
                i,
                message_clone,
                modulus_clone,
                exponent_clone,
                stats_clone,
            )
        }));
    }
    
    // Wait for all threads (in practice, they run until one finds a perfect cube)
    for handle in handles {
        handle.join().unwrap()?;
    }
    
    Ok(())
}