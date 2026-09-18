use std::error::Error;

#[derive(clap::Args, Clone)]
pub struct ShallengeArgs {
    /// Username for the challenge
    #[arg(long)]
    pub username: String,
    /// Target hash to beat (hex)
    #[arg(long)]
    pub target_hash: String,
}

impl ShallengeArgs {
    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        let Self {
            username,
            target_hash,
        } = self;

        crate::args::validate_hex_string(target_hash)?;
        if target_hash.len() != 64 {
            return Err("target hash must contain exactly 32 bytes (64 hex digits)".into());
        }
        if username.is_empty() {
            return Err("username cannot be empty".into());
        }
        if username.len() > logic::modes::shallenge::MAX_USERNAME_LEN {
            return Err(format!(
                        "username length {} exceeds max {} (preimage is fixed at {} bytes: username + '/' + nonce)",
                        username.len(),
                        logic::modes::shallenge::MAX_USERNAME_LEN,
                        logic::modes::shallenge::PREIMAGE_LEN
                    ).into());
        }

        Ok(())
    }
    pub fn details(&self) -> crate::args::CommandDetails {
        let Self {
            username,
            target_hash,
        } = self;
        crate::args::CommandDetails {
            prefix_len: username.len(),
            suffix_len: 0,
            cpu_threads: None,

            description: format!(
                "Starting shallenge for username '{}' with target hash '{}'",
                username, target_hash
            ),
        }
    }
}
