use crate::args::pattern::PatternArgs;
use clap::Args;
use clap::ValueEnum;
use std::path::PathBuf;

#[derive(ValueEnum, Clone, Copy)]
pub enum RsaPssSource {
    Salt,
    Message,
}

#[derive(Args, Clone)]
pub struct RsaPssArgs {
    #[command(flatten)]
    pub pattern: PatternArgs,
    #[arg(long)]
    pub key: PathBuf,
    #[arg(long)]
    pub message: PathBuf,
    #[arg(long, default_value = "sha256", value_parser = ["sha256"])]
    pub hash: String,
    #[arg(long, default_value_t = 32)]
    pub salt_length: usize,
    #[arg(long, value_enum, default_value = "salt")]
    pub search_source: RsaPssSource,
    #[arg(long, requires = "nonce_length")]
    pub nonce_offset: Option<usize>,
    #[arg(long, requires = "nonce_offset")]
    pub nonce_length: Option<usize>,
    #[arg(long)]
    pub fixed_salt_hex: Option<String>,
}

impl RsaPssArgs {
    pub fn config(&self, workers: usize) -> Result<crate::modes::rsa_pss::PssSearch, String> {
        use crate::modes::rsa_pss::PssSource;
        if self.hash != "sha256" {
            return Err("only SHA-256 is supported".into());
        }
        let source = match self.search_source {
            RsaPssSource::Salt => {
                if self.nonce_offset.is_some()
                    || self.nonce_length.is_some()
                    || self.fixed_salt_hex.is_some()
                {
                    return Err("salt search requires a fixed message and a varying salt".into());
                }
                PssSource::Salt {
                    length: self.salt_length,
                }
            }
            RsaPssSource::Message => {
                let fixed_salt = self
                    .fixed_salt_hex
                    .as_ref()
                    .map(|hex| {
                        hex::decode(hex)
                            .map_err(|_| "fixed salt must be complete hexadecimal bytes without 0x")
                    })
                    .transpose()?;
                if fixed_salt
                    .as_ref()
                    .is_some_and(|salt| salt.len() != self.salt_length)
                {
                    return Err("fixed salt length must equal --salt-length".into());
                }
                PssSource::Message {
                    offset: self
                        .nonce_offset
                        .ok_or("message search requires --nonce-offset")?,
                    length: self
                        .nonce_length
                        .ok_or("message search requires --nonce-length")?,
                    fixed_salt,
                    salt_length: self.salt_length,
                }
            }
        };
        Ok(crate::modes::rsa_pss::PssSearch {
            key: self.key.clone(),
            message: self.message.clone(),
            source,
            prefix: self.pattern.prefix.clone(),
            suffix: self.pattern.suffix.clone(),
            workers: workers,
        })
    }
}

impl RsaPssArgs {
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.config(1)?.validate()?;
        Ok(())
    }
    pub fn details(&self) -> crate::args::CommandDetails {
        self.pattern.details("Searching raw RSA-PSS signatures")
    }
}
