//! Shared application test mechanics; algorithms and expected bytes stay mode-owned.
use std::{path::PathBuf, process::Command};

pub fn artifacts(variable: &str, mode: &str) -> PathBuf {
    std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("VANITY_METAL_BUNDLES")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/metal")
                })
                .join(mode)
        })
}

pub fn command(directory: &std::path::Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_vanity-miner"));
    command.arg("--metal-artifacts").arg(directory);
    command
}

#[cfg(any(feature = "bitcoin", feature = "ethereum", feature = "solana"))]
pub mod address {
    use logic::search::{candidate_result::BatchResult, vanity::BytePattern, xoroshiro::BatchSeed};
    use std::path::Path;

    pub trait Engine: Sized {
        fn load(path: &Path, capacity: u32, group: usize, audit: bool) -> Result<Self, String>;
        fn evaluate(
            &mut self,
            seed: &BatchSeed,
            pattern: &BytePattern,
            start: u64,
            count: u32,
        ) -> Result<BatchResult, String>;
        fn launches(&self) -> u64;
    }
    macro_rules! engine {
        ($feature:literal, $mode:ident, $name:ident) => {
            #[cfg(feature = $feature)]
            impl Engine for vanity_miner::modes::$mode::metal::$name {
                fn load(
                    path: &Path,
                    capacity: u32,
                    group: usize,
                    audit: bool,
                ) -> Result<Self, String> {
                    Self::load(path, capacity, group, audit)
                }
                fn evaluate(
                    &mut self,
                    seed: &BatchSeed,
                    pattern: &BytePattern,
                    start: u64,
                    count: u32,
                ) -> Result<BatchResult, String> {
                    self.evaluate(seed, pattern, &[], start, count)
                }
                fn launches(&self) -> u64 {
                    self.launches
                }
            }
        };
    }
    engine!("bitcoin", bitcoin, BitcoinTransport);
    engine!("ethereum", ethereum, EthereumTransport);
    engine!("solana", solana, SolanaTransport);

    pub struct Cases<'a> {
        pub seed: u64,
        pub patterns: [BytePattern; 4],
        pub address: &'a [u8],
        pub winner_lane: u32,
        pub check_next_batch: bool,
        pub extra_groups: &'a [usize],
    }

    pub fn check<E: Engine>(path: &Path, cases: Cases<'_>) {
        let mut engine = E::load(path, 257, 64, true).unwrap();
        for count in [1, 31, 32, 33, 65, 129, 257] {
            for (seed, width, start) in [
                (cases.seed, 32, 0),
                (u64::MAX, 33, 31),
                (0, 1, u64::MAX - 256),
            ] {
                for (index, pattern) in cases.patterns.iter().enumerate() {
                    let result = engine
                        .evaluate(&BatchSeed { seed, width }, pattern, start, count)
                        .unwrap();
                    if index == 0 {
                        assert_eq!(result.matches, count);
                    }
                    if index == 1 {
                        assert_eq!(result.matches, 0);
                    }
                }
            }
        }
        let seed = BatchSeed {
            seed: cases.seed,
            width: 32,
        };
        let exact = BytePattern::new(cases.address, cases.address).unwrap();
        let winner = engine.evaluate(&seed, &exact, 0, 33).unwrap();
        assert_eq!((winner.matches, winner.lane), (1, cases.winner_lane));
        if cases.check_next_batch {
            assert_eq!(engine.evaluate(&seed, &exact, 33, 33).unwrap().matches, 0);
        }
        let mut conflict = exact;
        conflict.suffix[0] ^= 1;
        assert_eq!(engine.evaluate(&seed, &conflict, 0, 33).unwrap().matches, 0);
        let empty = BytePattern::new(&[], &[]).unwrap();
        for width in [0, u64::from(u32::MAX) + 1] {
            assert_eq!(
                engine
                    .evaluate(&BatchSeed { seed: 0, width }, &empty, 0, 33)
                    .err()
                    .as_deref(),
                Some("device candidate evaluation failed")
            );
        }
        for pattern in [
            BytePattern {
                prefix_len: 65,
                ..empty
            },
            BytePattern {
                suffix_len: 65,
                ..empty
            },
        ] {
            assert_eq!(
                engine.evaluate(&seed, &pattern, 0, 33).err().as_deref(),
                Some("device candidate evaluation failed")
            );
        }
        let before = engine.launches();
        for (start, count) in [(0, 0), (0, 258), (u64::MAX, 2)] {
            assert!(engine.evaluate(&seed, &empty, start, count).is_err());
        }
        assert_eq!(engine.launches(), before);
        for &group in cases.extra_groups {
            let mut grouped = E::load(path, 257, group, true).unwrap();
            assert_eq!(
                grouped.evaluate(&seed, &empty, 0, 257).unwrap().matches,
                257
            );
            let winner = grouped.evaluate(&seed, &exact, 0, 33).unwrap();
            assert_eq!((winner.matches, winner.lane), (1, cases.winner_lane));
        }
        let mut engine = E::load(path, 33, 32, false).unwrap();
        let winner = engine.evaluate(&seed, &exact, 0, 33).unwrap();
        assert_eq!((winner.matches, winner.lane), (1, cases.winner_lane));
        assert_eq!(engine.evaluate(&seed, &conflict, 0, 33).unwrap().matches, 0);
    }

    pub fn cli(path: &Path, command: &str, seed: u64, prefix: &str, suffix: &str) -> String {
        let output = super::command(path)
            .args([
                "--batches",
                "2",
                "--batch-size",
                "33",
                "--seed",
                &seed.to_string(),
                "--verify",
                command,
                "--prefix",
                prefix,
                "--suffix",
                suffix,
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("66 total keys"), "{stdout}");
        stdout
    }
}
