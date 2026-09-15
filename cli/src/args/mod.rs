#[cfg(all(test, feature = "bitcoin"))]
use crate::modes::bitcoin::args::BitcoinArgs;
#[cfg(all(test, feature = "ethereum"))]
use crate::modes::ethereum::args::EthereumArgs;
#[cfg(all(test, feature = "shallenge"))]
use crate::modes::shallenge::args::ShallengeArgs;
#[cfg(all(test, feature = "solana"))]
use crate::modes::solana::args::SolanaArgs;
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
    SolanaVanity(crate::modes::solana::args::SolanaArgs),
    /// Generate Bitcoin vanity address (bech32)
    #[cfg(feature = "bitcoin")]
    BitcoinVanity(crate::modes::bitcoin::args::BitcoinArgs),
    /// Generate Ethereum vanity address (hex)
    #[cfg(feature = "ethereum")]
    EthereumVanity(crate::modes::ethereum::args::EthereumArgs),
    /// Find better shallenge nonce
    #[cfg(feature = "shallenge")]
    Shallenge(crate::modes::shallenge::args::ShallengeArgs),
    /// Run on-device self-test (validates PTX codegen against CPU expectations)
    #[cfg(feature = "self_test_support")]
    SelfTest,
}

impl Command {
    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        match self {
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(args) => args.validate(),
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(args) => args.validate(),
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(args) => args.validate(),
            #[cfg(feature = "p256-signature")]
            Self::P256SignatureVanity(args) => args.validate(),
            #[cfg(feature = "solana")]
            Self::SolanaVanity(args) => args.validate(),
            #[cfg(feature = "bitcoin")]
            Self::BitcoinVanity(args) => args.validate(),
            #[cfg(feature = "ethereum")]
            Self::EthereumVanity(args) => args.validate(),
            #[cfg(feature = "shallenge")]
            Self::Shallenge(args) => args.validate(),
            #[cfg(feature = "self_test_support")]
            Self::SelfTest => Ok(()),
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
                    Command::SolanaVanity(SolanaArgs { prefix, suffix }) => (prefix, suffix),
                    #[cfg(feature = "bitcoin")]
                    Command::BitcoinVanity(BitcoinArgs { prefix, suffix }) => (prefix, suffix),
                    #[cfg(feature = "ethereum")]
                    Command::EthereumVanity(EthereumArgs { prefix, suffix }) => (prefix, suffix),
                    #[allow(unreachable_patterns)]
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
            Command::Shallenge(ShallengeArgs {
                username,
                target_hash,
            }) => {
                assert_eq!(username, "alice");
                assert_eq!(target_hash, hash);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("unexpected command"),
        }
        assert!(Cli::try_parse_from(["vanity-miner", "shallenge", "alice", &hash]).is_err());
        assert!(Cli::try_parse_from(["vanity-miner", "shallenge", "--username", "alice"]).is_err());
    }
}

/// Startup metadata shared by reporters and backend selection.
pub struct CommandDetails {
    pub prefix_len: usize,
    pub suffix_len: usize,
    pub description: String,
    #[cfg_attr(any(feature = "gpu", feature = "cumetal"), allow(dead_code))]
    pub cpu_threads: Option<usize>,
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub cuda_module: Option<&'static str>,
}

impl Command {
    pub fn details(&self) -> CommandDetails {
        match self {
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(args) => args.details(),
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(args) => args.details(),
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(args) => args.details(),
            #[cfg(feature = "p256-signature")]
            Self::P256SignatureVanity(args) => args.details(),
            #[cfg(feature = "solana")]
            Self::SolanaVanity(args) => args.details(),
            #[cfg(feature = "bitcoin")]
            Self::BitcoinVanity(args) => args.details(),
            #[cfg(feature = "ethereum")]
            Self::EthereumVanity(args) => args.details(),
            #[cfg(feature = "shallenge")]
            Self::Shallenge(args) => args.details(),
            #[cfg(feature = "self_test_support")]
            Self::SelfTest => CommandDetails {
                prefix_len: 0,
                suffix_len: 0,
                cpu_threads: None,
                cuda_module: None,
                description: "Running on-device self-test".into(),
            },
        }
    }
}

#[cfg(feature = "crypto-cli")]
pub(crate) mod pattern;

#[cfg(any(feature = "ethereum", feature = "shallenge"))]
pub fn validate_hex_string(hex_string: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    match hex::decode(hex_string) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Invalid hex string: {}", e).into()),
    }
}
