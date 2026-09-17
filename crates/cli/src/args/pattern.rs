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

impl PatternArgs {
    pub fn details(
        &self,
        description: &str,
        cuda_module: &'static str,
    ) -> crate::args::CommandDetails {
        crate::args::CommandDetails {
            prefix_len: self.prefix.len(),
            suffix_len: self.suffix.len(),
            cpu_threads: self.threads,
            description: description.into(),
            cuda_module: Some(cuda_module),
        }
    }
}
