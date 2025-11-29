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
    SolanaVanity {
        /// Prefix to search for
        prefix: String,
        /// Suffix to search for
        suffix: String,
    },
    /// Generate Bitcoin vanity address (bech32)
    BitcoinVanity {
        /// Prefix to search for
        prefix: String,
        /// Suffix to search for
        suffix: String,
    },
    /// Generate Ethereum vanity address (hex)
    EthereumVanity {
        /// Prefix to search for (hex, without 0x)
        prefix: String,
        /// Suffix to search for (hex, without 0x)
        suffix: String,
    },
    /// Find better shallenge nonce
    Shallenge {
        /// Username for the challenge
        username: String,
        /// Target hash to beat (hex)
        target_hash: String,
    },
}

impl Command {
    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        match self {
            Command::SolanaVanity { prefix, suffix } => {
                if !prefix.is_empty() {
                    common::validate_base58_string(prefix)?;
                }
                if !suffix.is_empty() {
                    common::validate_base58_string(suffix)?;
                }
            }
            Command::BitcoinVanity { prefix, suffix } => {
                if !prefix.is_empty() {
                    common::validate_bech32_string(prefix)?;
                }
                if !suffix.is_empty() {
                    common::validate_bech32_string(suffix)?;
                }
            }
            Command::EthereumVanity { prefix, suffix } => {
                if !prefix.is_empty() {
                    common::validate_hex_string(prefix)?;
                }
                if !suffix.is_empty() {
                    common::validate_hex_string(suffix)?;
                }
            }
            Command::Shallenge { target_hash, .. } => {
                common::validate_hex_string(target_hash)?;
            }
        }
        Ok(())
    }

    pub fn prefix_len(&self) -> usize {
        match self {
            Command::SolanaVanity { prefix, .. } => prefix.len(),
            Command::BitcoinVanity { prefix, .. } => prefix.len(),
            Command::EthereumVanity { prefix, .. } => prefix.len(),
            Command::Shallenge { username, .. } => username.len(),
        }
    }

    pub fn suffix_len(&self) -> usize {
        match self {
            Command::SolanaVanity { suffix, .. } => suffix.len(),
            Command::BitcoinVanity { suffix, .. } => suffix.len(),
            Command::EthereumVanity { suffix, .. } => suffix.len(),
            Command::Shallenge { .. } => 0,
        }
    }

    pub fn description(&self) -> String {
        match self {
            Command::SolanaVanity { prefix, suffix } => {
                format!("Searching for solana vanity key with prefix '{}' and suffix '{}'", prefix, suffix)
            }
            Command::BitcoinVanity { prefix, suffix } => {
                format!("Searching for bitcoin vanity key with prefix '{}' and suffix '{}'", prefix, suffix)
            }
            Command::EthereumVanity { prefix, suffix } => {
                format!("Searching for ethereum vanity key with prefix '{}' and suffix '{}'", prefix, suffix)
            }
            Command::Shallenge { username, target_hash } => {
                format!("Starting shallenge for username '{}' with target hash '{}'", username, target_hash)
            }
        }
    }
}
