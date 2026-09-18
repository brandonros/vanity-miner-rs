use crate::args::pattern::PatternArgs;
use clap::Args;
use clap::ValueEnum;
use std::path::PathBuf;

#[derive(ValueEnum, Clone, Copy)]
pub enum P256Source {
    Message,
    Ephemeral,
}
#[derive(ValueEnum, Clone, Copy)]
pub enum SignatureTarget {
    Raw,
    R,
    S,
}
#[derive(ValueEnum, Clone, Copy)]
pub enum SForm {
    Low,
    High,
    Either,
}

#[derive(Args, Clone)]
pub struct P256SignatureArgs {
    #[command(flatten)]
    pub pattern: PatternArgs,
    #[arg(long)]
    pub key: PathBuf,
    #[arg(long)]
    pub message: PathBuf,
    #[arg(long, default_value = "sha256", value_parser = ["sha256"])]
    pub hash: String,
    #[arg(long, value_enum, default_value = "message")]
    pub search_source: P256Source,
    #[arg(long, requires = "nonce_length")]
    pub nonce_offset: Option<usize>,
    #[arg(long, requires = "nonce_offset")]
    pub nonce_length: Option<usize>,
    #[arg(long, value_enum, default_value = "raw")]
    pub target: SignatureTarget,
    #[arg(long, value_enum, default_value = "low")]
    pub s_form: SForm,
}

impl P256SignatureArgs {
    pub fn config(
        &self,
        workers: usize,
    ) -> Result<crate::modes::p256_signature::SignatureSearch, String> {
        use crate::modes::p256_signature::SearchSource;
        use logic::crypto::p256::signatures::{SForm as Form, SignatureTarget as Target};
        if self.hash != "sha256" {
            return Err("only SHA-256 is supported".into());
        }
        let source = match self.search_source {
            P256Source::Message => SearchSource::Message {
                offset: self
                    .nonce_offset
                    .ok_or("message search requires --nonce-offset")?,
                length: self
                    .nonce_length
                    .ok_or("message search requires --nonce-length")?,
            },
            P256Source::Ephemeral => {
                if self.nonce_offset.is_some() || self.nonce_length.is_some() {
                    return Err("ephemeral search does not modify a message window".into());
                }
                SearchSource::Ephemeral
            }
        };
        Ok(crate::modes::p256_signature::SignatureSearch {
            key: self.key.clone(),
            message: self.message.clone(),
            source,
            prefix: self.pattern.prefix.clone(),
            suffix: self.pattern.suffix.clone(),
            target: match self.target {
                SignatureTarget::Raw => Target::Raw,
                SignatureTarget::R => Target::R,
                SignatureTarget::S => Target::S,
            },
            s_form: match self.s_form {
                SForm::Low => Form::Low,
                SForm::High => Form::High,
                SForm::Either => Form::Either,
            },
            workers: self.pattern.threads.unwrap_or(workers),
        })
    }
}

impl P256SignatureArgs {
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.config(1)?.validate()?;
        Ok(())
    }
    pub fn details(&self) -> crate::args::CommandDetails {
        self.pattern.details("Searching NIST P-256 signatures")
    }
}
