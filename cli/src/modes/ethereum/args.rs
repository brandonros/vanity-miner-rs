use std::error::Error;

#[derive(clap::Args, Clone)]
pub struct EthereumArgs {
    /// Prefix to search for (hex, without 0x)
    #[arg(long, default_value = "")]
    pub prefix: String,
    /// Suffix to search for (hex, without 0x)
    #[arg(long, default_value = "")]
    pub suffix: String,
}

impl EthereumArgs {
    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let Self { prefix, suffix } = self;

        if !prefix.is_empty() {
            crate::args::validate_hex_string(prefix)?;
        }
        if !suffix.is_empty() {
            crate::args::validate_hex_string(suffix)?;
        }

        Ok(())
    }
    pub fn details(&self) -> crate::args::CommandDetails {
        let Self { prefix, suffix } = self;
        crate::args::CommandDetails {
            prefix_len: prefix.len(),
            suffix_len: suffix.len(),
            cpu_threads: None,
            cuda_module: Some("ethereum"),
            description: format!(
                "Searching for ethereum vanity key with prefix '{}' and suffix '{}'",
                prefix, suffix
            ),
        }
    }
}
