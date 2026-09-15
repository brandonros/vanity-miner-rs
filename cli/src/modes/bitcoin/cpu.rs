use crate::common::GlobalStats;
use std::error::Error;
use std::sync::Arc;

use crate::common::spawn_cpu_workers;
use rand::Rng as _;

struct WorkerData {
    prefix_bytes: Vec<u8>,
    suffix_bytes: Vec<u8>,
    global_stats: Arc<GlobalStats>,
}

fn worker(
    thread_id: usize,
    data: Arc<WorkerData>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut rng = rand::thread_rng();

    println!("[CPU-{thread_id}] Starting CPU bitcoin vanity worker thread");

    while !cancelled.load(std::sync::atomic::Ordering::Relaxed) {
        let rng_seed: u64 = rng.r#gen();

        let request = logic::modes::bitcoin_vanity::BitcoinVanityKeyRequest {
            prefix: &data.prefix_bytes,
            suffix: &data.suffix_bytes,
            thread_idx: thread_id,
            rng_seed,
        };

        let result = logic::modes::bitcoin_vanity::generate_and_check_bitcoin_vanity_key(&request);

        data.global_stats.add_launch(1);

        if result.matches {
            let encoded_public_key_str =
                std::str::from_utf8(&result.encoded_public_key[0..result.encoded_len])
                    .unwrap_or("invalid_utf8");
            let mut encoded_private_key = [0u8; 64];
            let encoded_len = logic::modes::bitcoin_vanity::private_key_to_wif(
                &result.private_key,
                true,
                false,
                &mut encoded_private_key,
            );
            let encoded_private_key_str =
                std::str::from_utf8(&encoded_private_key[0..encoded_len]).unwrap_or("invalid_utf8");

            println!("[CPU-{thread_id}] Vanity match: rng_seed = {rng_seed}");
            println!("[CPU-{thread_id}] Vanity match: thread_idx = {thread_id}");
            println!(
                "[CPU-{thread_id}] Vanity match: encoded_public_key = {encoded_public_key_str}"
            );
            println!(
                "[CPU-{thread_id}] Vanity match: public_key = {}",
                hex::encode(result.public_key)
            );
            println!(
                "[CPU-{thread_id}] Vanity match: public_key_hash = {}",
                hex::encode(result.public_key_hash)
            );
            println!(
                "[CPU-{thread_id}] Vanity match: private_key = {}",
                hex::encode(result.private_key)
            );
            println!("[CPU-{thread_id}] Vanity match: wallet = {encoded_private_key_str}");

            data.global_stats.add_matches(1);
        }
    }
    Ok(())
}

pub fn run(
    num_threads: usize,
    prefix: String,
    suffix: String,
    global_stats: Arc<GlobalStats>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!(
        "Starting CPU bitcoin vanity mode with {} threads",
        num_threads
    );

    let data = Arc::new(WorkerData {
        prefix_bytes: prefix.as_bytes().to_vec(),
        suffix_bytes: suffix.as_bytes().to_vec(),
        global_stats,
    });

    spawn_cpu_workers(num_threads, data, worker)
}
