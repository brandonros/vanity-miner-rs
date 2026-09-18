use crate::runner::progress::GlobalStats;
use std::fmt::Write as _;
use std::sync::Arc;

use crate::runner::{
    progress::print_verified,
    session::{SearchControl, run_controlled},
    workers,
};
use rand::Rng as _;

struct WorkerData {
    prefix_bytes: Vec<u8>,
    suffix_bytes: Vec<u8>,
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

        let request = logic::modes::solana::SolanaVanityKeyRequest {
            prefix: &data.prefix_bytes,
            suffix: &data.suffix_bytes,
            thread_idx: thread_id,
            rng_seed,
        };

        let result = logic::modes::solana::generate_and_check_solana_vanity_key(&request);

        data.global_stats.add_launch(1);

        if result.matches && cancelled.claim_verified_winner() {
            let encoded_str =
                std::str::from_utf8(&result.encoded_public_key[0..result.encoded_len])
                    .unwrap_or("invalid_utf8");

            let mut record = String::new();
            writeln!(
                &mut record,
                "[CPU-{thread_id}] Vanity match: rng_seed = {rng_seed}"
            )
            .unwrap();
            writeln!(
                &mut record,
                "[CPU-{thread_id}] Vanity match: thread_idx = {thread_id}"
            )
            .unwrap();
            writeln!(
                &mut record,
                "[CPU-{thread_id}] Vanity match: encoded_public_key = {encoded_str}"
            )
            .unwrap();
            writeln!(
                &mut record,
                "[CPU-{thread_id}] Vanity match: public_key = {}",
                hex::encode(result.public_key)
            )
            .unwrap();
            writeln!(
                &mut record,
                "[CPU-{thread_id}] Vanity match: private_key = {}",
                hex::encode(result.private_key)
            )
            .unwrap();
            writeln!(
                &mut record,
                "[CPU-{thread_id}] Vanity match: wallet = {}",
                hex::encode([result.private_key, result.public_key].concat())
            )
            .unwrap();

            return Ok(Some(record));
        }
    }
    Ok(None)
}

pub fn run(
    args: &super::args::SolanaArgs,
    workers: usize,
    global_stats: Arc<GlobalStats>,
    exit_on_first_match: bool,
) -> crate::runner::RunResult {
    let data = WorkerData {
        prefix_bytes: args.prefix.as_bytes().to_vec(),
        suffix_bytes: args.suffix.as_bytes().to_vec(),
        global_stats: global_stats.clone(),
    };

    run_controlled(global_stats, "keys", exit_on_first_match, |control| {
        let winner = workers::search(workers, &control, |id| worker(id, &data, &control))?;
        if let Some(record) = winner {
            print_verified(&control, record)?;
            Ok(true)
        } else {
            Ok(false)
        }
    })
}
