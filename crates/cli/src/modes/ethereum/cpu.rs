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

        let request = logic::modes::ethereum::EthereumVanityKeyRequest {
            prefix: &data.prefix_bytes,
            suffix: &data.suffix_bytes,
            thread_idx: thread_id,
            rng_seed,
        };

        let result = logic::modes::ethereum::generate_and_check_ethereum_vanity_key(&request);

        data.global_stats.add_launch(1);

        if result.matches && cancelled.claim_verified_winner() {
            let encoded_address_str = hex::encode(result.address);

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
                "[CPU-{thread_id}] Vanity match: address = 0x{encoded_address_str}"
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
                "[CPU-{thread_id}] Vanity match: private_key = 0x{}",
                hex::encode(result.private_key)
            )
            .unwrap();
            writeln!(
                &mut record,
                "[CPU-{thread_id}] Vanity match: wallet = 0x{}",
                hex::encode(result.private_key)
            )
            .unwrap();

            return Ok(Some(record));
        }
    }
    Ok(None)
}

pub fn run(
    args: &super::args::EthereumArgs,
    workers: usize,
    global_stats: Arc<GlobalStats>,
    exit_on_first_match: bool,
) -> crate::runner::RunResult {
    let prefix_bytes = hex::decode(&args.prefix)?;
    let suffix_bytes = hex::decode(&args.suffix)?;

    let data = WorkerData {
        prefix_bytes,
        suffix_bytes,
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
