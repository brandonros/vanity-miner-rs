use clap::Args;

#[derive(Args, Clone)]
pub struct PatternArgs {
    /// Case-insensitive hexadecimal prefix, without 0x; odd digit counts are valid
    #[arg(long, default_value = "")]
    pub prefix: String,
    /// Case-insensitive hexadecimal suffix, without 0x
    #[arg(long, default_value = "")]
    pub suffix: String,
    /// CPU worker count (default: available parallelism)
    #[arg(long)]
    pub threads: Option<usize>,
}
