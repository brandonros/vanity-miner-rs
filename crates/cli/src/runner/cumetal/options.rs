use clap::Args;
use std::path::PathBuf;

#[derive(Args, Clone)]
pub struct CumetalOptions {
    /// CuMetal package built from flake.lock; supplied by `nix develop .#cumetal`.
    #[arg(long, global = true, default_value = option_env!("VANITY_CUMETAL_ROOT"))]
    pub cumetal_root: Option<PathBuf>,
    /// Rust-CUDA PTX file, or a directory of separately compiled PTX modules.
    #[arg(long, global = true)]
    pub ptx: Option<PathBuf>,
    /// Stop after this many kernel launches; omitted means keep searching.
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
}
