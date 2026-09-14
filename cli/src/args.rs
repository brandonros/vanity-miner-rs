use clap::{Parser, Subcommand};
use std::error::Error;

#[derive(Parser)]
#[command(name = "vanity-miner")]
#[command(about = "GPU-accelerated vanity address generator for multiple blockchains")]
pub struct Cli {
    #[cfg(feature = "cumetal")]
    #[command(flatten)]
    pub cumetal: crate::runner::CumetalOptions,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Clone)]
// Feature subsets can leave only the public CLI names ending in `Vanity`.
#[allow(clippy::enum_variant_names)]
pub enum Command {
    /// Construct a matching RSA-2048 modulus and export its key pair
    #[cfg(all(feature = "rsa-modulus", not(feature = "cumetal")))]
    RsaModulusVanity(crate::crypto_args::RsaModulusArgs),
    /// Search raw RSA-PSS signatures over salts or a message window
    #[cfg(all(feature = "rsa-pss", not(feature = "cumetal")))]
    RsaPssSignatureVanity(crate::crypto_args::RsaPssArgs),
    /// Generate a matching NIST P-256 public point and private key
    #[cfg(all(feature = "p256-public-key", not(feature = "cumetal")))]
    P256PublicKeyVanity(crate::crypto_args::P256PublicArgs),
    /// Search P-256 signatures over a message window or secret ephemeral nonces
    #[cfg(all(feature = "p256-signature", not(feature = "cumetal")))]
    P256SignatureVanity(crate::crypto_args::P256SignatureArgs),
    /// Generate Solana vanity address (base58)
    #[cfg(feature = "solana")]
    SolanaVanity {
        /// Prefix to search for
        prefix: String,
        /// Suffix to search for
        suffix: String,
    },
    /// Generate Bitcoin vanity address (bech32)
    #[cfg(feature = "bitcoin")]
    BitcoinVanity {
        /// Prefix to search for
        prefix: String,
        /// Suffix to search for
        suffix: String,
    },
    /// Generate Ethereum vanity address (hex)
    #[cfg(feature = "ethereum")]
    EthereumVanity {
        /// Prefix to search for (hex, without 0x)
        prefix: String,
        /// Suffix to search for (hex, without 0x)
        suffix: String,
    },
    /// Find better shallenge nonce
    #[cfg(feature = "shallenge")]
    Shallenge {
        /// Username for the challenge
        username: String,
        /// Target hash to beat (hex)
        target_hash: String,
    },
    /// Run on-device self-test (validates PTX codegen against CPU expectations)
    #[cfg(feature = "self_test")]
    SelfTest,
}

impl Command {
    #[cfg(not(feature = "gpu"))]
    pub fn cpu_threads(&self, default: usize) -> usize {
        match self {
            #[cfg(all(feature = "rsa-modulus", not(feature = "cumetal")))]
            Self::RsaModulusVanity(args) => args.pattern.threads.unwrap_or(default),
            #[cfg(all(feature = "rsa-pss", not(feature = "cumetal")))]
            Self::RsaPssSignatureVanity(args) => args.pattern.threads.unwrap_or(default),
            #[cfg(all(feature = "p256-public-key", not(feature = "cumetal")))]
            Self::P256PublicKeyVanity(args) => args.pattern.threads.unwrap_or(default),
            #[cfg(all(feature = "p256-signature", not(feature = "cumetal")))]
            Self::P256SignatureVanity(args) => args.pattern.threads.unwrap_or(default),
            #[allow(unreachable_patterns)]
            _ => default,
        }
    }

    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        match self {
            #[cfg(all(feature = "rsa-modulus", not(feature = "cumetal")))]
            Self::RsaModulusVanity(args) => {
                args.config(1)?.validate()?;
            }
            #[cfg(all(feature = "rsa-pss", not(feature = "cumetal")))]
            Self::RsaPssSignatureVanity(args) => {
                args.config(1)?.validate()?;
            }
            #[cfg(all(feature = "p256-public-key", not(feature = "cumetal")))]
            Self::P256PublicKeyVanity(args) => args.config(1).validate()?,
            #[cfg(all(feature = "p256-signature", not(feature = "cumetal")))]
            Self::P256SignatureVanity(args) => args.config(1)?.validate()?,
            #[cfg(feature = "solana")]
            Command::SolanaVanity { prefix, suffix } => {
                if !prefix.is_empty() {
                    crate::common::validate_base58_string(prefix)?;
                }
                if !suffix.is_empty() {
                    crate::common::validate_base58_string(suffix)?;
                }
            }
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity { prefix, suffix } => {
                if !prefix.is_empty() {
                    crate::common::validate_bech32_string(prefix)?;
                }
                if !suffix.is_empty() {
                    crate::common::validate_bech32_string(suffix)?;
                }
            }
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity { prefix, suffix } => {
                if !prefix.is_empty() {
                    crate::common::validate_hex_string(prefix)?;
                }
                if !suffix.is_empty() {
                    crate::common::validate_hex_string(suffix)?;
                }
            }
            #[cfg(feature = "shallenge")]
            Command::Shallenge {
                username,
                target_hash,
            } => {
                crate::common::validate_hex_string(target_hash)?;
                if username.is_empty() {
                    return Err("username cannot be empty".into());
                }
                if username.len() > logic::MAX_USERNAME_LEN {
                    return Err(format!(
                        "username length {} exceeds max {} (preimage is fixed at {} bytes: username + '/' + nonce)",
                        username.len(),
                        logic::MAX_USERNAME_LEN,
                        logic::PREIMAGE_LEN
                    ).into());
                }
            }
            #[cfg(feature = "self_test")]
            Command::SelfTest => {}
        }
        Ok(())
    }

    pub fn prefix_len(&self) -> usize {
        match self {
            #[cfg(all(feature = "rsa-modulus", not(feature = "cumetal")))]
            Self::RsaModulusVanity(args) => args.pattern.prefix.len(),
            #[cfg(all(feature = "rsa-pss", not(feature = "cumetal")))]
            Self::RsaPssSignatureVanity(args) => args.pattern.prefix.len(),
            #[cfg(all(feature = "p256-public-key", not(feature = "cumetal")))]
            Self::P256PublicKeyVanity(args) => args.pattern.prefix.len(),
            #[cfg(all(feature = "p256-signature", not(feature = "cumetal")))]
            Self::P256SignatureVanity(args) => args.pattern.prefix.len(),
            #[cfg(feature = "solana")]
            Command::SolanaVanity { prefix, .. } => prefix.len(),
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity { prefix, .. } => prefix.len(),
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity { prefix, .. } => prefix.len(),
            #[cfg(feature = "shallenge")]
            Command::Shallenge { username, .. } => username.len(),
            #[cfg(feature = "self_test")]
            Command::SelfTest => 0,
        }
    }

    pub fn suffix_len(&self) -> usize {
        match self {
            #[cfg(all(feature = "rsa-modulus", not(feature = "cumetal")))]
            Self::RsaModulusVanity(args) => args.pattern.suffix.len(),
            #[cfg(all(feature = "rsa-pss", not(feature = "cumetal")))]
            Self::RsaPssSignatureVanity(args) => args.pattern.suffix.len(),
            #[cfg(all(feature = "p256-public-key", not(feature = "cumetal")))]
            Self::P256PublicKeyVanity(args) => args.pattern.suffix.len(),
            #[cfg(all(feature = "p256-signature", not(feature = "cumetal")))]
            Self::P256SignatureVanity(args) => args.pattern.suffix.len(),
            #[cfg(feature = "solana")]
            Command::SolanaVanity { suffix, .. } => suffix.len(),
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity { suffix, .. } => suffix.len(),
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity { suffix, .. } => suffix.len(),
            #[cfg(feature = "shallenge")]
            Command::Shallenge { .. } => 0,
            #[cfg(feature = "self_test")]
            Command::SelfTest => 0,
        }
    }

    pub fn description(&self) -> String {
        match self {
            #[cfg(all(feature = "rsa-modulus", not(feature = "cumetal")))]
            Self::RsaModulusVanity(_) => "Constructing an RSA-2048 vanity modulus".into(),
            #[cfg(all(feature = "rsa-pss", not(feature = "cumetal")))]
            Self::RsaPssSignatureVanity(_) => "Searching raw RSA-PSS signatures".into(),
            #[cfg(all(feature = "p256-public-key", not(feature = "cumetal")))]
            Self::P256PublicKeyVanity(_) => "Searching NIST P-256 public points".into(),
            #[cfg(all(feature = "p256-signature", not(feature = "cumetal")))]
            Self::P256SignatureVanity(_) => "Searching NIST P-256 signatures".into(),
            #[cfg(feature = "solana")]
            Command::SolanaVanity { prefix, suffix } => {
                format!(
                    "Searching for solana vanity key with prefix '{}' and suffix '{}'",
                    prefix, suffix
                )
            }
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity { prefix, suffix } => {
                format!(
                    "Searching for bitcoin vanity key with prefix '{}' and suffix '{}'",
                    prefix, suffix
                )
            }
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity { prefix, suffix } => {
                format!(
                    "Searching for ethereum vanity key with prefix '{}' and suffix '{}'",
                    prefix, suffix
                )
            }
            #[cfg(feature = "shallenge")]
            Command::Shallenge {
                username,
                target_hash,
            } => {
                format!(
                    "Starting shallenge for username '{}' with target hash '{}'",
                    username, target_hash
                )
            }
            #[cfg(feature = "self_test")]
            Command::SelfTest => "Running on-device self-test".to_string(),
        }
    }
}
