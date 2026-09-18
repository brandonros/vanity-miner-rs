//! Warm Solana throughput sweep. Run correctness tests before this benchmark.
use logic::search::{vanity::BytePattern, xoroshiro::BatchSeed};
use std::{
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};
use vanity_miner::runner::metal::transport::SolanaTransport;

fn main() -> Result<(), String> {
    let artifacts = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "target/metal/solana".into());
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifacts.join("kernel.build.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    // A valid, selective Base58 prefix exercises a normal production search.
    // Abort if it matches so winner handling cannot contaminate this timing.
    let pattern = BytePattern::new(b"zzzzzzzz", b"")?;
    for count in [256_u32, 1024, 4096] {
        let seed = BatchSeed {
            seed: 583437459223573146,
            width: u64::from(count),
        };
        let cpu_count = 1024u64;
        let start = Instant::now();
        for counter in 0..cpu_count {
            let result = black_box(logic::modes::solana::candidate(
                black_box(&seed),
                counter,
                black_box(&pattern),
            ));
            if result.status != 0 {
                return Err("unexpected CPU benchmark result".into());
            }
        }
        let cpu = start.elapsed().as_secs_f64();
        for group in [32, 64, 128] {
            let mut engine = SolanaTransport::load(&artifacts, count, group, false)?;
            for batch in 0..3 {
                engine.evaluate(&seed, &pattern, batch * u64::from(count), count)?;
            }
            let before = (
                engine.dispatch_time,
                engine.upload_time,
                engine.download_time,
                engine.gpu_time,
                engine.gpu_timed_launches,
            );
            let start = Instant::now();
            let mut repeats = 0;
            while start.elapsed() < Duration::from_secs(1) || repeats < 5 {
                let result =
                    engine.evaluate(&seed, &pattern, (3 + repeats) * u64::from(count), count)?;
                if result.matches != 0 || result.errors != 0 {
                    return Err("unexpected GPU benchmark result".into());
                }
                repeats += 1;
            }
            let wall = start.elapsed().as_secs_f64();
            let total = u64::from(count) * repeats;
            let gpu = (engine.gpu_timed_launches - before.4 == repeats)
                .then(|| (engine.gpu_time - before.3).as_secs_f64());
            println!(
                "{}",
                serde_json::json!({
                    "batch_size":count,"threads_per_group":group,"candidates":total,
                    "warmup_batches":3,"timed_batches":repeats,"audit":false,
                    "host_seconds":wall,"dispatch_seconds":(engine.dispatch_time-before.0).as_secs_f64(),
                    "upload_seconds":(engine.upload_time-before.1).as_secs_f64(),
                    "download_seconds":(engine.download_time-before.2).as_secs_f64(),"gpu_seconds":gpu,
                    "candidates_per_second":total as f64/wall,
                    "cpu_single_thread_candidates":cpu_count,"cpu_single_thread_seconds":cpu,
                    "cpu_candidates_per_second":cpu_count as f64/cpu,
                    "library_seconds":engine.load_stages.library.as_secs_f64(),
                    "pipeline_seconds":engine.load_stages.pipeline.as_secs_f64(),
                    "allocation_seconds":engine.allocation_time.as_secs_f64(),
                    "metallib_sha256":manifest["artifacts"]["kernel.metallib"],
                    "compiler_sha256":manifest["compiler_sha256"]
                })
            );
        }
    }
    Ok(())
}
