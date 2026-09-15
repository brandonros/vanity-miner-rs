use std::error::Error;

pub fn validate_base58_string(base58_string: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let invalid_characters = ["l", "I", "0", "O"];
    for invalid_character in invalid_characters {
        if base58_string.contains(invalid_character) {
            return Err(format!(
                "base58 string contains invalid character: {}",
                invalid_character
            )
            .into());
        }
    }
    Ok(())
}

#[derive(clap::Args, Clone)]
pub struct SolanaArgs {
    /// Prefix to search for
    #[arg(long, default_value = "")]
    pub prefix: String,
    /// Suffix to search for
    #[arg(long, default_value = "")]
    pub suffix: String,
}

impl SolanaArgs {
    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let Self { prefix, suffix } = self;

        if !prefix.is_empty() {
            validate_base58_string(prefix)?;
        }
        if !suffix.is_empty() {
            validate_base58_string(suffix)?;
        }

        Ok(())
    }
    pub fn details(&self) -> crate::args::CommandDetails {
        let Self { prefix, suffix } = self;
        crate::args::CommandDetails {
            prefix_len: prefix.len(),
            suffix_len: suffix.len(),
            cpu_threads: None,
            cuda_module: Some("solana"),
            continuous_candidates: false,
            description: format!(
                "Searching for solana vanity key with prefix '{}' and suffix '{}'",
                prefix, suffix
            ),
        }
    }
}
