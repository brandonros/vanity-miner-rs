use crate::modes::shallenge::shared_best_hash::SharedBestHash;
use crate::runner::progress::GlobalStats;
use std::fmt::Write as _;
use std::sync::{Arc, RwLock};

use crate::runner::{
    progress::print_verified,
    session::{SearchControl, run_controlled},
    workers,
};
use rand::Rng as _;

struct WorkerData {
    username: String,
    shared_best_hash: Arc<RwLock<SharedBestHash>>,
    global_stats: Arc<GlobalStats>,
}

fn worker(
    thread_id: usize,
    data: &WorkerData,
    cancelled: &SearchControl,
) -> Result<Option<String>, String> {
    let mut rng = rand::thread_rng();

    while !cancelled.stopped() {
        let rng_seed: u64 = rng.r#gen();

        // Get the current best hash (with minimal lock time)
        // Use unwrap_or_else to recover data even if lock is poisoned
        let current_target = {
            let best_hash_guard = data
                .shared_best_hash
                .read()
                .unwrap_or_else(|e| e.into_inner());
            best_hash_guard.get_current()
        };

        // Create the request with the current best target
        let request = logic::modes::shallenge::ShallengeRequest {
            username: data.username.as_bytes(),
            username_len: data.username.len(),
            target_hash: &current_target,
            thread_idx: thread_id,
            rng_seed,
        };

        let result = logic::modes::shallenge::generate_and_check_shallenge(&request);

        data.global_stats.add_launch(1);

        if result.is_better {
            let nonce_string =
                std::str::from_utf8(&result.nonce[0..result.nonce_len]).unwrap_or("invalid_utf8");

            // Try to update the global best hash
            let was_global_best = {
                let mut best_hash_guard = data
                    .shared_best_hash
                    .write()
                    .unwrap_or_else(|e| e.into_inner());
                if result.hash < best_hash_guard.get_current() && cancelled.claim_verified_winner()
                {
                    best_hash_guard.update_if_better(result.hash)
                } else {
                    false
                }
            };

            if was_global_best {
                let mut record = String::new();
                writeln!(
                    &mut record,
                    "[CPU-{}] NEW GLOBAL BEST found: thread_idx = {}",
                    thread_id, thread_id
                )
                .unwrap();
                writeln!(
                    &mut record,
                    "[CPU-{}] NEW GLOBAL BEST hash: {}",
                    thread_id,
                    hex::encode(result.hash)
                )
                .unwrap();
                writeln!(
                    &mut record,
                    "[CPU-{}] NEW GLOBAL BEST nonce: {}",
                    thread_id, nonce_string
                )
                .unwrap();
                writeln!(
                    &mut record,
                    "[CPU-{}] Challenge string: {}/{}",
                    thread_id, data.username, nonce_string
                )
                .unwrap();

                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

pub fn run(
    args: &super::args::ShallengeArgs,
    workers: usize,
    global_stats: Arc<GlobalStats>,
    exit_on_first_match: bool,
) -> crate::runner::RunResult {
    // Convert Vec<u8> to [u8; 32] for the initial target
    let initial_target: [u8; 32] = hex::decode(&args.target_hash)?
        .try_into()
        .map_err(|_| "target hash must contain exactly 32 bytes")?;

    // Create shared state for the best hash found so far
    let shared_best_hash = Arc::new(RwLock::new(SharedBestHash::new(initial_target)));

    let data = WorkerData {
        username: args.username.clone(),
        shared_best_hash,
        global_stats: global_stats.clone(),
    };

    run_controlled(global_stats, "nonces", exit_on_first_match, |control| {
        let winner = workers::search(workers, &control, |id| worker(id, &data, &control))?;
        if let Some(record) = winner {
            print_verified(&control, record)?;
            Ok(true)
        } else {
            Ok(false)
        }
    })
}
