#![cfg(all(feature = "metal", feature = "bitcoin", target_os = "macos"))]
use logic::search::{vanity::BytePattern, xoroshiro::BatchSeed};
use std::{path::PathBuf, process::Command};
use vanity_miner::runner::metal::transport::BitcoinTransport;

fn artifacts() -> PathBuf {
    std::env::var_os("VANITY_METAL_BITCOIN_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/metal/bitcoin")
        })
}

#[test]
#[ignore = "build scripts/build-metal.py --mode bitcoin first; requires Apple GPU"]
fn candidates_patterns_winners_and_errors_match_cpu() {
    let mut engine = BitcoinTransport::load(&artifacts(), 257, 64, true).unwrap();
    for count in [1, 31, 32, 33, 65, 129, 257] {
        for (seed, width, start) in [
            (10088153575472065218, 32, 0),
            (u64::MAX, 33, 31),
            (0, 1, u64::MAX - 256),
        ] {
            for pattern in [
                BytePattern::new(&[], &[]).unwrap(),
                BytePattern::new(&[b'q'; 43], &[]).unwrap(),
                BytePattern::new(b"bc1qg", b"6m").unwrap(),
                BytePattern::new(&[], b"q").unwrap(),
            ] {
                let result = engine
                    .evaluate(&BatchSeed { seed, width }, &pattern, start, count)
                    .unwrap();
                if pattern.prefix_len == 0 && pattern.suffix_len == 0 {
                    assert_eq!(result.matches, count);
                }
                if pattern.prefix_len == 43 {
                    assert_eq!(result.matches, 0);
                }
            }
        }
    }
    let seed = BatchSeed {
        seed: 10088153575472065218,
        width: 32,
    };
    let address = b"bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m".to_vec();
    // Full-width overlapping prefix/suffix, a unique known lane, then a miss.
    let exact = BytePattern::new(&address, &address).unwrap();
    let winner = engine.evaluate(&seed, &exact, 0, 33).unwrap();
    assert_eq!((winner.matches, winner.lane), (1, 0));
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
    let before = engine.launches;
    for (start, count) in [(0, 0), (0, 258), (u64::MAX, 2)] {
        assert!(engine.evaluate(&seed, &empty, start, count).is_err());
    }
    assert_eq!(engine.launches, before);

    // Winner verification is mandatory even when returning all lanes is disabled.
    let mut engine = BitcoinTransport::load(&artifacts(), 33, 32, false).unwrap();
    let winner = engine.evaluate(&seed, &exact, 0, 33).unwrap();
    assert_eq!((winner.matches, winner.lane), (1, 0));
    assert_eq!(engine.evaluate(&seed, &conflict, 0, 33).unwrap().matches, 0);
}

#[test]
#[ignore = "build scripts/build-metal.py --mode bitcoin first; requires Apple GPU"]
fn bounded_cli_prints_verified_winners_and_completes_misses() {
    for (prefix, suffix, matches) in [
        ("", "", true),
        ("bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m", "", true),
        ("bc1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq", "", false),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_vanity-miner"))
            .args([
                "--metal-artifacts",
                artifacts().to_str().unwrap(),
                "--batches",
                "2",
                "--batch-size",
                "33",
                "--seed",
                "10088153575472065218",
                "--verify",
                "bitcoin-vanity",
                "--prefix",
                prefix,
                "--suffix",
                suffix,
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let stdout = String::from_utf8(result.stdout).unwrap();
        assert!(stdout.contains("66 total keys"), "{stdout}");
        assert_eq!(stdout.contains("[bitcoin] wallet="), matches, "{stdout}");
        assert_eq!(stdout.contains("[bitcoin] address="), matches, "{stdout}");
        if prefix.starts_with("bc1qg") {
            assert!(
                stdout.contains("[bitcoin] address=bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m")
            );
        }
    }
}
