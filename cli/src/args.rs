use clap::{Parser, Subcommand};
use std::error::Error;

#[derive(Parser)]
#[command(name = "vanity-miner")]
#[command(about = "GPU-accelerated vanity address generator for multiple blockchains")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Clone)]
pub enum Command {
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
    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        match self {
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
            Command::Shallenge { target_hash, .. } => {
                crate::common::validate_hex_string(target_hash)?;
            }
            #[cfg(feature = "self_test")]
            Command::SelfTest => {}
        }
        Ok(())
    }

    pub fn prefix_len(&self) -> usize {
        match self {
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
            #[cfg(feature = "solana")]
            Command::SolanaVanity { prefix, suffix } => {
                format!("Searching for solana vanity key with prefix '{}' and suffix '{}'", prefix, suffix)
            }
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity { prefix, suffix } => {
                format!("Searching for bitcoin vanity key with prefix '{}' and suffix '{}'", prefix, suffix)
            }
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity { prefix, suffix } => {
                format!("Searching for ethereum vanity key with prefix '{}' and suffix '{}'", prefix, suffix)
            }
            #[cfg(feature = "shallenge")]
            Command::Shallenge { username, target_hash } => {
                format!("Starting shallenge for username '{}' with target hash '{}'", username, target_hash)
            }
            #[cfg(feature = "self_test")]
            Command::SelfTest => {
                "Running on-device self-test".to_string()
            }
        }
    }
}
