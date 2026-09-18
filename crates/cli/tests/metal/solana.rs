#![cfg(all(feature = "metal", feature = "solana", target_os = "macos"))]
use logic::search::vanity::BytePattern;
use std::path::PathBuf;
use vanity_miner::runner::metal::transport::SolanaTransport;

fn artifacts() -> PathBuf {
    super::support::artifacts("VANITY_METAL_SOLANA_ARTIFACTS", "solana")
}

#[test]
#[ignore = "build scripts/build-metal.py --mode solana first; requires Apple GPU"]
fn candidates_patterns_winners_and_errors_match_cpu() {
    super::support::address::check::<SolanaTransport>(
        &artifacts(),
        super::support::address::Cases {
            seed: 583437459223573146,
            patterns: [
                BytePattern::new(&[], &[]).unwrap(),
                BytePattern::new(&[b'z'; 45], &[]).unwrap(),
                BytePattern::new(b"aaa", b"NFC").unwrap(),
                BytePattern::new(&[], b"C").unwrap(),
            ],
            address: logic::test_vectors::SOLANA_ADDRESS.as_bytes(),
            winner_lane: 3,
            check_next_batch: true,
            extra_groups: &[32, 128],
        },
    );
}

#[test]
#[ignore = "build scripts/build-metal.py --mode solana first; requires Apple GPU"]
fn bounded_cli_prints_verified_winners_and_completes_misses() {
    for (prefix, suffix, matches) in [
        ("", "", true),
        (logic::test_vectors::SOLANA_ADDRESS, "", true),
        ("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", "", false),
    ] {
        let stdout = super::support::address::cli(
            &artifacts(),
            "solana-vanity",
            583437459223573146,
            prefix,
            suffix,
        );
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
