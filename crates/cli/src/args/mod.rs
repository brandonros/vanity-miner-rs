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
    /// Stop successfully after printing the first fully verified match.
    #[arg(long, global = true)]
    pub exit_on_first_match: bool,
    /// CPU worker count (default: available parallelism)
    #[cfg(not(feature = "metal"))]
    #[arg(long, global = true)]
    pub threads: Option<std::num::NonZeroUsize>,
    #[cfg(feature = "metal")]
    #[command(flatten)]
    pub metal: crate::runner::metal::MetalOptions,
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
    /// Run known-answer self-tests on the selected backend
    #[cfg(feature = "self_test_support")]
    SelfTest(crate::modes::self_test::args::SelfTestArgs),
}

impl Command {
    pub fn validate(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        match *self {
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(ref args) => args.validate(),
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(ref args) => args.validate(),
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(ref args) => args.validate(),
            #[cfg(feature = "p256-signature")]
            Self::P256SignatureVanity(ref args) => args.validate(),
            #[cfg(feature = "solana")]
            Self::SolanaVanity(ref args) => args.validate(),
            #[cfg(feature = "bitcoin")]
            Self::BitcoinVanity(ref args) => args.validate(),
            #[cfg(feature = "ethereum")]
            Self::EthereumVanity(ref args) => args.validate(),
            #[cfg(feature = "shallenge")]
            Self::Shallenge(ref args) => args.validate(),
            #[cfg(feature = "self_test_support")]
            Self::SelfTest(ref args) => args.selected().map(|_| ()).map_err(Into::into),
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
}

impl Command {
    pub fn details(&self) -> CommandDetails {
        match *self {
            #[cfg(feature = "rsa-modulus")]
            Self::RsaModulusVanity(ref args) => args.details(),
            #[cfg(feature = "rsa-pss")]
            Self::RsaPssSignatureVanity(ref args) => args.details(),
            #[cfg(feature = "p256-public-key")]
            Self::P256PublicKeyVanity(ref args) => args.details(),
            #[cfg(feature = "p256-signature")]
            Self::P256SignatureVanity(ref args) => args.details(),
            #[cfg(feature = "solana")]
            Self::SolanaVanity(ref args) => args.details(),
            #[cfg(feature = "bitcoin")]
            Self::BitcoinVanity(ref args) => args.details(),
            #[cfg(feature = "ethereum")]
            Self::EthereumVanity(ref args) => args.details(),
            #[cfg(feature = "shallenge")]
            Self::Shallenge(ref args) => args.details(),
            #[cfg(feature = "self_test_support")]
            Self::SelfTest(_) => CommandDetails {
                prefix_len: 0,
                suffix_len: 0,
                description: "Running self-tests".into(),
            },
        }
    }
}

#[cfg(any(
    feature = "p256-public-key",
    feature = "p256-signature",
    feature = "rsa-modulus",
    feature = "rsa-pss"
))]
pub(crate) mod pattern;

#[cfg(any(feature = "ethereum", feature = "shallenge"))]
pub fn validate_hex_string(hex_string: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    match hex::decode(hex_string) {
        Ok(_) => Ok(()),
        Err(e) => Err(format!("Invalid hex string: {}", e).into()),
    }
}

#[cfg(test)]
mod first_match_tests {
    use super::*;

    #[test]
    fn flag_is_global_and_all_search_modes_default_to_continuous() {
        let commands: &[&[&str]] = &[
            #[cfg(feature = "shallenge")]
            &[
                "shallenge",
                "--username",
                "miner",
                "--target-hash",
                "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            ],
            #[cfg(feature = "ethereum")]
            &["ethereum-vanity"],
            #[cfg(feature = "bitcoin")]
            &["bitcoin-vanity"],
            #[cfg(feature = "solana")]
            &["solana-vanity"],
            #[cfg(feature = "rsa-modulus")]
            &["rsa-modulus-vanity"],
            #[cfg(feature = "p256-public-key")]
            &["p256-public-key-vanity"],
            #[cfg(feature = "p256-signature")]
            &[
                "p256-signature-vanity",
                "--key",
                "key.pem",
                "--message",
                "message.bin",
                "--search-source",
                "ephemeral",
            ],
            #[cfg(feature = "rsa-pss")]
            &[
                "rsa-pss-signature-vanity",
                "--key",
                "key.pem",
                "--message",
                "message.bin",
            ],
        ];
        for command in commands {
            let mut args = vec!["vanity-miner"];
            args.extend_from_slice(command);
            assert!(!Cli::try_parse_from(&args).unwrap().exit_on_first_match);
            let mut threads = args.clone();
            threads.extend(["--threads", "2"]);
            #[cfg(not(feature = "metal"))]
            {
                assert_eq!(
                    Cli::try_parse_from(&threads)
                        .unwrap()
                        .threads
                        .unwrap()
                        .get(),
                    2
                );
                *threads.last_mut().unwrap() = "0";
                assert!(Cli::try_parse_from(&threads).is_err());
            }
            #[cfg(feature = "metal")]
            assert!(Cli::try_parse_from(&threads).is_err());

            args.insert(1, "--exit-on-first-match");
            assert!(Cli::try_parse_from(&args).unwrap().exit_on_first_match);
            args.remove(1);
            args.push("--exit-on-first-match");
            assert!(Cli::try_parse_from(&args).unwrap().exit_on_first_match);
        }
    }
}
