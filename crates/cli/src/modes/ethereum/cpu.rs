use crate::runner::progress::GlobalStats;
use std::error::Error;
use std::sync::Arc;

use crate::runner::workers::cpu::spawn_cpu_workers;
use rand::Rng as _;

struct WorkerData {
    prefix_bytes: Vec<u8>,
    suffix_bytes: Vec<u8>,
    global_stats: Arc<GlobalStats>,
}

fn worker(
    thread_id: usize,
    data: Arc<WorkerData>,
    cancelled: Arc<crate::runner::session::SearchControl>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut rng = rand::thread_rng();

    println!("[CPU-{thread_id}] Starting CPU ethereum vanity worker thread");

    while !cancelled.stopped() {
        let rng_seed: u64 = rng.r#gen();

        let request = logic::modes::ethereum::EthereumVanityKeyRequest {
            prefix: &data.prefix_bytes,
            suffix: &data.suffix_bytes,
            thread_idx: thread_id,
            rng_seed,
        };

        let result = logic::modes::ethereum::generate_and_check_ethereum_vanity_key(&request);

        data.global_stats.add_launch(1);

        if result.matches {
            let encoded_address_str = hex::encode(result.address);

            println!("[CPU-{thread_id}] Vanity match: rng_seed = {rng_seed}");
            println!("[CPU-{thread_id}] Vanity match: thread_idx = {thread_id}");
            println!("[CPU-{thread_id}] Vanity match: address = 0x{encoded_address_str}");
            println!(
                "[CPU-{thread_id}] Vanity match: public_key = {}",
                hex::encode(result.public_key)
            );
            println!(
                "[CPU-{thread_id}] Vanity match: private_key = 0x{}",
                hex::encode(result.private_key)
            );
            println!(
                "[CPU-{thread_id}] Vanity match: wallet = 0x{}",
                hex::encode(result.private_key)
            );

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
    // Ethereum uses hex-encoded prefix/suffix
    let prefix_bytes = hex::decode(&prefix)?;
    let suffix_bytes = hex::decode(&suffix)?;

    println!(
        "Starting CPU ethereum vanity mode with {} threads",
        num_threads
    );

    let data = Arc::new(WorkerData {
        prefix_bytes,
        suffix_bytes,
        global_stats,
    });

    spawn_cpu_workers(num_threads, data, worker)
}
