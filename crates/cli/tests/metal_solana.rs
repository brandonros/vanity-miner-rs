#![cfg(all(feature = "metal", feature = "solana", target_os = "macos"))]
use logic::search::{vanity::BytePattern, xoroshiro::BatchSeed};
use std::{path::PathBuf, process::Command};
use vanity_miner::runner::metal::transport::SolanaTransport;

fn artifacts() -> PathBuf {
    std::env::var_os("VANITY_METAL_SOLANA_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/metal/solana")
        })
}

#[test]
#[ignore = "build scripts/build-metal.py --mode solana first; requires Apple GPU"]
fn candidates_patterns_winners_and_errors_match_cpu() {
    let mut engine = SolanaTransport::load(&artifacts(), 257, 64, true).unwrap();
    for count in [1, 31, 32, 33, 65, 129, 257] {
        for (seed, width, start) in [
            (583437459223573146, 32, 0),
            (u64::MAX, 33, 31),
            (0, 1, u64::MAX - 256),
        ] {
            for pattern in [
                BytePattern::new(&[], &[]).unwrap(),
                BytePattern::new(&[b'z'; 45], &[]).unwrap(),
                BytePattern::new(b"aaa", b"NFC").unwrap(),
                BytePattern::new(&[], b"C").unwrap(),
            ] {
                let result = engine
                    .evaluate(&BatchSeed { seed, width }, &pattern, start, count)
                    .unwrap();
                if pattern.prefix_len == 0 && pattern.suffix_len == 0 {
                    assert_eq!(result.matches, count);
                }
                if pattern.prefix_len == 45 {
                    assert_eq!(result.matches, 0);
                }
            }
        }
    }
    let seed = BatchSeed {
        seed: 583437459223573146,
        width: 32,
    };
    let address = b"aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC".to_vec();
    // Full-width overlapping prefix/suffix, a unique known lane, then a miss.
    let exact = BytePattern::new(&address, &address).unwrap();
    let winner = engine.evaluate(&seed, &exact, 0, 33).unwrap();
    assert_eq!((winner.matches, winner.lane), (1, 3));
    // The next batch must not repeat the known lane from the first batch.
    assert_eq!(engine.evaluate(&seed, &exact, 33, 33).unwrap().matches, 0);
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

    for group in [32, 128] {
        let mut grouped = SolanaTransport::load(&artifacts(), 257, group, true).unwrap();
        assert_eq!(
            grouped.evaluate(&seed, &empty, 0, 257).unwrap().matches,
            257
        );
        let winner = grouped.evaluate(&seed, &exact, 0, 33).unwrap();
        assert_eq!((winner.matches, winner.lane), (1, 3));
    }

    // Winner verification is mandatory even when returning all lanes is disabled.
    let mut engine = SolanaTransport::load(&artifacts(), 33, 32, false).unwrap();
    let winner = engine.evaluate(&seed, &exact, 0, 33).unwrap();
    assert_eq!((winner.matches, winner.lane), (1, 3));
    assert_eq!(engine.evaluate(&seed, &conflict, 0, 33).unwrap().matches, 0);
}

#[test]
#[ignore = "build scripts/build-metal.py --mode solana first; requires Apple GPU"]
fn bounded_cli_prints_verified_winners_and_completes_misses() {
    for (prefix, suffix, matches) in [
        ("", "", true),
        ("aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC", "", true),
        ("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", "", false),
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
                "583437459223573146",
                "--verify",
                "solana-vanity",
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
        assert_eq!(stdout.contains("[solana] wallet="), matches, "{stdout}");
        assert_eq!(stdout.contains("[solana] address="), matches, "{stdout}");
        if prefix.starts_with("aaa") {
            assert_eq!(stdout.matches("[solana] address=").count(), 1, "{stdout}");
            assert!(
                stdout.contains("[solana] address=aaatgciWHhvVra6u4znVSfSqqJszUcpDDFEEKrPjNFC")
            );
        }
    }
}
