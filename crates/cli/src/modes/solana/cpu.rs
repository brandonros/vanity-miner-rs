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

    println!("[CPU-{thread_id}] Starting CPU solana vanity worker thread");

    while !cancelled.stopped() {
        let rng_seed: u64 = rng.r#gen();

        let request = logic::modes::solana::SolanaVanityKeyRequest {
            prefix: &data.prefix_bytes,
            suffix: &data.suffix_bytes,
            thread_idx: thread_id,
            rng_seed,
        };

        let result = logic::modes::solana::generate_and_check_solana_vanity_key(&request);

        data.global_stats.add_launch(1);

        if result.matches && (!cancelled.exit_on_first_match() || cancelled.claim_verified_winner())
        {
            let encoded_str =
                std::str::from_utf8(&result.encoded_public_key[0..result.encoded_len])
                    .unwrap_or("invalid_utf8");

            println!("[CPU-{thread_id}] Vanity match: rng_seed = {rng_seed}");
            println!("[CPU-{thread_id}] Vanity match: thread_idx = {thread_id}");
            println!("[CPU-{thread_id}] Vanity match: encoded_public_key = {encoded_str}");
            println!(
                "[CPU-{thread_id}] Vanity match: public_key = {}",
                hex::encode(result.public_key)
            );
            println!(
                "[CPU-{thread_id}] Vanity match: private_key = {}",
                hex::encode(result.private_key)
            );
            println!(
                "[CPU-{thread_id}] Vanity match: wallet = {}",
                hex::encode([result.private_key, result.public_key].concat())
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
    exit_on_first_match: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!(
        "Starting CPU solana vanity mode with {} threads",
        num_threads
    );

    let control = Arc::new(crate::runner::session::SearchControl::with_stats(
        global_stats.clone(),
    ));
    if exit_on_first_match {
        control.set_exit_on_first_match();
    }
    let data = Arc::new(WorkerData {
        prefix_bytes: prefix.as_bytes().to_vec(),
        suffix_bytes: suffix.as_bytes().to_vec(),
        global_stats,
    });

    spawn_cpu_workers(num_threads, data, control, worker)
}
