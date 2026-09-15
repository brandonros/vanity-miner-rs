use clap::Args;
use std::path::PathBuf;

#[derive(Args, Clone)]
pub struct CumetalOptions {
    /// CuMetal driver library (libcumetal.dylib).
    #[arg(long, global = true, default_value = "libcumetal.dylib")]
    pub cumetal_library: PathBuf,
    /// Rust-CUDA PTX file, or a directory of separately compiled PTX modules.
    #[arg(long, global = true)]
    pub ptx: Option<PathBuf>,
    /// Directory of precompiled ENTRY.metal files and their ABI sidecars.
    #[arg(long, global = true)]
    pub module_dir: Option<PathBuf>,
    #[arg(long, global = true, default_value = "cumetalc")]
    pub cumetalc: PathBuf,
    /// Stop after this many batches (four kernel stages per RSA modulus batch); omitted means keep searching.
    #[arg(long, global=true, value_parser=clap::value_parser!(u64).range(1..))]
    pub batches: Option<u64>,
    #[arg(long, global=true, default_value_t=32, value_parser=clap::value_parser!(u32).range(1..=1024))]
    pub threads_per_block: u32,
    #[arg(long, global=true, default_value_t=1, value_parser=clap::value_parser!(u32).range(1..=65535))]
    pub blocks: u32,
    /// Starting deterministic seed; subsequent batches increment it.
    #[arg(long, global = true)]
    pub seed: Option<u64>,
    /// Compare every candidate and the match count with the CPU reference.
    #[arg(long, global = true)]
    pub verify: bool,
    #[cfg(feature = "self_test_support")]
    /// Report selected self-test slots; each containing mode kernel still runs in full.
    #[arg(long, global=true, value_parser=clap::value_parser!(u32).range(0..logic::self_test::SELF_TEST_NUM_CHECKS as i64))]
    pub self_test_slot: Vec<u32>,
}
