//! Explicit deterministic benchmark; correctness is covered by --verify/tests.
use logic::search::xoroshiro::BatchSeed;
use std::{
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};
use vanity_miner::runner::metal::transport::ShallengeTransport;

fn main() -> Result<(), String> {
    let artifacts = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| "target/metal/shallenge".into());
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifacts.join("kernel.build.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    for count in [32_u32, 256, 4096, 16384, 65536] {
        let seed = BatchSeed {
            seed: 12345,
            width: u64::from(count),
        };
        let mut engine = ShallengeTransport::load(&artifacts, count, 64, false)?;
        for batch in 0..3 {
            engine.evaluate(
                &seed,
                &[0; 32],
                b"brandonros",
                batch * u64::from(count),
                count,
            )?;
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
        while start.elapsed() < Duration::from_secs(1) || repeats < 20 {
            let result = engine.evaluate(
                &seed,
                &[0; 32],
                b"brandonros",
                (3 + repeats) * u64::from(count),
                count,
            )?;
            if result.matches != 0 || result.errors != 0 {
                return Err("unexpected benchmark result".into());
            }
            repeats += 1;
        }
        let wall = start.elapsed().as_secs_f64();
        let dispatched = (engine.dispatch_time - before.0).as_secs_f64();
        let upload = (engine.upload_time - before.1).as_secs_f64();
        let download = (engine.download_time - before.2).as_secs_f64();
        let gpu = (engine.gpu_timed_launches - before.4 == repeats)
            .then(|| (engine.gpu_time - before.3).as_secs_f64());
        let start = Instant::now();
        for counter in 3 * u64::from(count)..(3 + repeats) * u64::from(count) {
            black_box(logic::modes::shallenge::candidate(
                black_box(&seed),
                counter,
                black_box(&[0; 32]),
                black_box(b"brandonros"),
            ));
        }
        let cpu = start.elapsed().as_secs_f64();
        let total = u64::from(count) * repeats;
        println!(
            "{}",
            serde_json::json!({"batch_size":count,"threads_per_group":64,"candidates":total,"warmup_batches":3,"timed_batches":repeats,"host_seconds":wall,"dispatch_seconds":dispatched,"upload_seconds":upload,"download_seconds":download,"gpu_seconds":gpu,"candidates_per_second":total as f64/wall,"cpu_single_thread_seconds":cpu,"cpu_candidates_per_second":total as f64/cpu,"audit":false,"load_seconds":engine.load_time.as_secs_f64(),"library_seconds":engine.load_stages.library.as_secs_f64(),"pipeline_seconds":engine.load_stages.pipeline.as_secs_f64(),"allocation_seconds":engine.allocation_time.as_secs_f64(),"metallib_sha256":manifest["artifacts"]["kernel.metallib"]})
        );
    }
    Ok(())
}
