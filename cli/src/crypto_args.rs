use clap::Args;
#[cfg(any(
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-pss"
))]
use clap::ValueEnum;
use std::path::PathBuf;

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
    /// Replace existing output files
    #[arg(long)]
    pub force: bool,
}

#[cfg(feature = "rsa-modulus")]
#[derive(Args, Clone)]
pub struct RsaModulusArgs {
    #[command(flatten)]
    pub pattern: PatternArgs,
    #[arg(long, default_value_t = 2048)]
    pub bits: usize,
    #[arg(long, default_value_t = 65537)]
    pub public_exponent: u32,
    #[arg(long, default_value = "constructive", value_parser = ["constructive"])]
    pub strategy: String,
    #[arg(long)]
    pub private_out: PathBuf,
    #[arg(long)]
    pub public_out: PathBuf,
}

#[cfg(feature = "rsa-modulus")]
impl RsaModulusArgs {
    pub fn config(
        &self,
        workers: usize,
    ) -> Result<vanity_miner::rsa_modulus::ModulusSearch, String> {
        if self.bits != 2048 || self.public_exponent != 65537 {
            return Err("RSA modulus search requires --bits 2048 --public-exponent 65537".into());
        }
        if self.strategy != "constructive" {
            return Err("only constructive RSA modulus search is supported".into());
        }
        Ok(vanity_miner::rsa_modulus::ModulusSearch {
            prefix: self.pattern.prefix.clone(),
            suffix: self.pattern.suffix.clone(),
            private_out: self.private_out.clone(),
            public_out: self.public_out.clone(),
            force: self.pattern.force,
            workers: self.pattern.threads.unwrap_or(workers),
        })
    }
}

#[cfg(feature = "p256-public-key")]
#[derive(ValueEnum, Clone, Copy)]
pub enum PublicTarget {
    X,
    Y,
    Xy,
    Uncompressed,
}
#[cfg(feature = "p256-public-key")]
#[derive(ValueEnum, Clone, Copy)]
pub enum PublicEncoding {
    Sec1,
    SpkiPem,
}

#[cfg(feature = "p256-public-key")]
#[derive(Args, Clone)]
pub struct P256PublicArgs {
    #[command(flatten)]
    pub pattern: PatternArgs,
    #[arg(long, value_enum, default_value = "xy")]
    pub target: PublicTarget,
    #[arg(long)]
    pub private_out: PathBuf,
    #[arg(long)]
    pub public_out: PathBuf,
    #[arg(long, value_enum, default_value = "sec1")]
    pub public_format: PublicEncoding,
}

#[cfg(feature = "p256-public-key")]
impl P256PublicArgs {
    pub fn config(&self, workers: usize) -> vanity_miner::p256_public::PublicKeySearch {
        use logic::p256_vanity::PublicTarget as Target;
        use vanity_miner::p256_public::PublicEncoding as Encoding;
        vanity_miner::p256_public::PublicKeySearch {
            prefix: self.pattern.prefix.clone(),
            suffix: self.pattern.suffix.clone(),
            target: match self.target {
                PublicTarget::X => Target::X,
                PublicTarget::Y => Target::Y,
                PublicTarget::Xy => Target::Xy,
                PublicTarget::Uncompressed => Target::Uncompressed,
            },
            private_out: self.private_out.clone(),
            public_out: self.public_out.clone(),
            public_encoding: match self.public_format {
                PublicEncoding::Sec1 => Encoding::Sec1,
                PublicEncoding::SpkiPem => Encoding::SpkiPem,
            },
            force: self.pattern.force,
            workers: self.pattern.threads.unwrap_or(workers),
        }
    }
}

#[cfg(feature = "p256-signature")]
#[derive(ValueEnum, Clone, Copy)]
pub enum P256Source {
    Message,
    Ephemeral,
}
#[cfg(feature = "p256-signature")]
#[derive(ValueEnum, Clone, Copy)]
pub enum SignatureTarget {
    Raw,
    R,
    S,
}
#[cfg(feature = "p256-signature")]
#[derive(ValueEnum, Clone, Copy)]
pub enum SForm {
    Low,
    High,
    Either,
}

#[cfg(feature = "p256-signature")]
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
    #[arg(long)]
    pub signature_out: PathBuf,
    #[arg(long)]
    pub message_out: Option<PathBuf>,
    #[arg(long)]
    pub der_out: Option<PathBuf>,
}

#[cfg(feature = "p256-signature")]
impl P256SignatureArgs {
    pub fn config(
        &self,
        workers: usize,
    ) -> Result<vanity_miner::p256_signature::SignatureSearch, String> {
        use logic::p256_vanity::signatures::{SForm as Form, SignatureTarget as Target};
        use vanity_miner::p256_signature::SearchSource;
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
        Ok(vanity_miner::p256_signature::SignatureSearch {
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
            signature_out: self.signature_out.clone(),
            message_out: self.message_out.clone(),
            der_out: self.der_out.clone(),
            force: self.pattern.force,
            workers: self.pattern.threads.unwrap_or(workers),
        })
    }
}

#[cfg(feature = "rsa-pss")]
#[derive(ValueEnum, Clone, Copy)]
pub enum RsaPssSource {
    Salt,
    Message,
}

#[cfg(feature = "rsa-pss")]
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
    #[arg(long)]
    pub signature_out: PathBuf,
    #[arg(long)]
    pub salt_out: PathBuf,
    #[arg(long)]
    pub message_out: Option<PathBuf>,
}

#[cfg(feature = "rsa-pss")]
impl RsaPssArgs {
    pub fn config(
        &self,
        workers: usize,
    ) -> Result<vanity_miner::rsa_pss_search::PssSearch, String> {
        use vanity_miner::rsa_pss_search::PssSource;
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
        Ok(vanity_miner::rsa_pss_search::PssSearch {
            key: self.key.clone(),
            message: self.message.clone(),
            source,
            prefix: self.pattern.prefix.clone(),
            suffix: self.pattern.suffix.clone(),
            signature_out: self.signature_out.clone(),
            salt_out: self.salt_out.clone(),
            message_out: self.message_out.clone(),
            force: self.pattern.force,
            workers: self.pattern.threads.unwrap_or(workers),
        })
    }
}
