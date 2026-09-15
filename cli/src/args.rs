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
    /// Continuously print matching RSA-2048 moduli and key pairs
    #[cfg(feature = "rsa-modulus")]
    RsaModulusVanity(crate::modes::rsa_modulus::args::RsaModulusArgs),
    /// Search raw RSA-PSS signatures over salts or a message window
    #[cfg(feature = "rsa-pss")]
    RsaPssSignatureVanity(crate::modes::rsa_pss::args::RsaPssArgs),
    /// Continuously print matching NIST P-256 public points and private keys
    #[cfg(feature = "p256-public-key")]
    P256PublicKeyVanity(crate::modes::p256_public_key::args::P256PublicArgs),
    /// Search P-256 signatures over a message window or secret ephemeral nonces
    #[cfg(feature = "p256-signature")]
    P256SignatureVanity(crate::modes::p256_signature::args::P256SignatureArgs),
    /// Generate Solana vanity address (base58)
    #[cfg(feature = "solana")]
    SolanaVanity {
        /// Prefix to search for
        #[arg(long, default_value = "")]
        prefix: String,
        /// Suffix to search for
        #[arg(long, default_value = "")]
        suffix: String,
    },
    /// Generate Bitcoin vanity address (bech32)
    #[cfg(feature = "bitcoin")]
    BitcoinVanity {
        /// Prefix to search for
        #[arg(long, default_value = "")]
        prefix: String,
        /// Suffix to search for
        #[arg(long, default_value = "")]
        suffix: String,
    },
    /// Generate Ethereum vanity address (hex)
    #[cfg(feature = "ethereum")]
    EthereumVanity {
        /// Prefix to search for (hex, without 0x)
        #[arg(long, default_value = "")]
        prefix: String,
        /// Suffix to search for (hex, without 0x)
        #[arg(long, default_value = "")]
        suffix: String,
    },
    /// Find better shallenge nonce
    #[cfg(feature = "shallenge")]
    Shallenge {
        /// Username for the challenge
        #[arg(long)]
        username: String,
        /// Target hash to beat (hex)
        #[arg(long)]
        target_hash: String,
    },
    /// Run on-device self-test (validates PTX codegen against CPU expectations)
    #[cfg(feature = "self_test_support")]
    SelfTest,
}

impl Command {
    #[cfg(not(any(feature = "gpu", feature = "cumetal")))]
    pub fn cpu_threads(&self, default: usize) -> usize {
        match self {
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(args) => args.pattern.threads.unwrap_or(default),
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(args) => args.pattern.threads.unwrap_or(default),
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(args) => args.pattern.threads.unwrap_or(default),
            #[cfg(feature = "p256-signature")]
            Self::P256SignatureVanity(args) => args.pattern.threads.unwrap_or(default),
            #[allow(unreachable_patterns)]
            _ => default,
        }
    }

    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        match self {
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(args) => {
                args.config(1)?.validate()?;
            }
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(args) => {
                args.config(1)?.validate()?;
            }
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(args) => args.config(1).validate()?,
            #[cfg(feature = "p256-signature")]
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
                    crate::common::validate_bech32_suffix(suffix)?;
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
                if username.len() > logic::modes::shallenge::MAX_USERNAME_LEN {
                    return Err(format!(
                        "username length {} exceeds max {} (preimage is fixed at {} bytes: username + '/' + nonce)",
                        username.len(),
                        logic::modes::shallenge::MAX_USERNAME_LEN,
                        logic::modes::shallenge::PREIMAGE_LEN
                    ).into());
                }
            }
            #[cfg(feature = "self_test_support")]
            Command::SelfTest => {}
        }
        Ok(())
    }

    pub fn prefix_len(&self) -> usize {
        match self {
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(args) => args.pattern.prefix.len(),
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(args) => args.pattern.prefix.len(),
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(args) => args.pattern.prefix.len(),
            #[cfg(feature = "p256-signature")]
            Self::P256SignatureVanity(args) => args.pattern.prefix.len(),
            #[cfg(feature = "solana")]
            Command::SolanaVanity { prefix, .. } => prefix.len(),
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity { prefix, .. } => prefix.len(),
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity { prefix, .. } => prefix.len(),
            #[cfg(feature = "shallenge")]
            Command::Shallenge { username, .. } => username.len(),
            #[cfg(feature = "self_test_support")]
            Command::SelfTest => 0,
        }
    }

    pub fn suffix_len(&self) -> usize {
        match self {
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(args) => args.pattern.suffix.len(),
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(args) => args.pattern.suffix.len(),
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(args) => args.pattern.suffix.len(),
            #[cfg(feature = "p256-signature")]
            Self::P256SignatureVanity(args) => args.pattern.suffix.len(),
            #[cfg(feature = "solana")]
            Command::SolanaVanity { suffix, .. } => suffix.len(),
            #[cfg(feature = "bitcoin")]
            Command::BitcoinVanity { suffix, .. } => suffix.len(),
            #[cfg(feature = "ethereum")]
            Command::EthereumVanity { suffix, .. } => suffix.len(),
            #[cfg(feature = "shallenge")]
            Command::Shallenge { .. } => 0,
            #[cfg(feature = "self_test_support")]
            Command::SelfTest => 0,
        }
    }

    pub fn description(&self) -> String {
        match self {
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(_) => "Constructing an RSA-2048 vanity modulus".into(),
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(_) => "Searching raw RSA-PSS signatures".into(),
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(_) => "Searching NIST P-256 public points".into(),
            #[cfg(feature = "p256-signature")]
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
            #[cfg(feature = "self_test_support")]
            Command::SelfTest => "Running on-device self-test".to_string(),
        }
    }
}

#[cfg(all(
    test,
    any(
        feature = "solana",
        feature = "bitcoin",
        feature = "ethereum",
        feature = "shallenge"
    )
))]
mod tests {
    use super::*;

    #[cfg(any(feature = "solana", feature = "bitcoin", feature = "ethereum"))]
    #[test]
    fn address_modes_use_named_optional_patterns() {
        let commands: &[&str] = &[
            #[cfg(feature = "solana")]
            "solana-vanity",
            #[cfg(feature = "bitcoin")]
            "bitcoin-vanity",
            #[cfg(feature = "ethereum")]
            "ethereum-vanity",
        ];
        for command in commands {
            let prefix = if *command == "bitcoin-vanity" {
                "bc1q"
            } else {
                "aa"
            };
            for (options, expected_prefix, expected_suffix) in [
                (vec!["--suffix", "ff", "--prefix", prefix], prefix, "ff"),
                (vec!["--prefix", prefix], prefix, ""),
                (vec!["--suffix", "ff"], "", "ff"),
                (vec![], "", ""),
            ] {
                let mut argv = vec!["vanity-miner", *command];
                argv.extend(options);
                let cli = Cli::try_parse_from(argv).unwrap();
                cli.command.validate().unwrap();
                let (prefix, suffix) = match cli.command {
                    #[cfg(feature = "solana")]
                    Command::SolanaVanity { prefix, suffix } => (prefix, suffix),
                    #[cfg(feature = "bitcoin")]
                    Command::BitcoinVanity { prefix, suffix } => (prefix, suffix),
                    #[cfg(feature = "ethereum")]
                    Command::EthereumVanity { prefix, suffix } => (prefix, suffix),
                    _ => panic!("unexpected command"),
                };
                assert_eq!(prefix, expected_prefix);
                assert_eq!(suffix, expected_suffix);
            }
            assert!(Cli::try_parse_from(["vanity-miner", command, "aa", ""]).is_err());
            assert!(Cli::try_parse_from(["vanity-miner", command, "--prefix"]).is_err());
        }
    }

    #[cfg(feature = "shallenge")]
    #[test]
    fn shallenge_requires_named_username_and_target() {
        let hash = "ff".repeat(32);
        let cli = Cli::try_parse_from([
            "vanity-miner",
            "shallenge",
            "--target-hash",
            &hash,
            "--username",
            "alice",
        ])
        .unwrap();
        cli.command.validate().unwrap();
        match cli.command {
            Command::Shallenge {
                username,
                target_hash,
            } => {
                assert_eq!(username, "alice");
                assert_eq!(target_hash, hash);
            }
            _ => panic!("unexpected command"),
        }
        assert!(Cli::try_parse_from(["vanity-miner", "shallenge", "alice", &hash]).is_err());
        assert!(Cli::try_parse_from(["vanity-miner", "shallenge", "--username", "alice"]).is_err());
    }
}

impl Command {
    #[cfg(feature = "gpu")]
    pub(crate) fn ptx_module(&self) -> &'static str {
        match self {
            #[cfg(feature = "solana")]
            Self::SolanaVanity { .. } => "solana",
            #[cfg(feature = "bitcoin")]
            Self::BitcoinVanity { .. } => "bitcoin",
            #[cfg(feature = "ethereum")]
            Self::EthereumVanity { .. } => "ethereum",
            #[cfg(feature = "shallenge")]
            Self::Shallenge { .. } => "shallenge",
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(_) => "p256_public_key",
            #[cfg(feature = "p256-signature")]
            Self::P256SignatureVanity(_) => "p256_signature",
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(_) => "rsa_pss",
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(_) => "rsa_modulus",
            #[cfg(feature = "self_test_support")]
            Self::SelfTest => unreachable!("self-tests select their own modules"),
        }
    }
}
