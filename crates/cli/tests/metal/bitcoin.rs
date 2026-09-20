#![cfg(all(feature = "metal", feature = "bitcoin", target_os = "macos"))]
use logic::search::vanity::BytePattern;
use std::path::PathBuf;
use vanity_miner::modes::bitcoin::metal::BitcoinTransport;

fn artifacts() -> PathBuf {
    super::support::artifacts("VANITY_METAL_BITCOIN_ARTIFACTS", "bitcoin")
}

#[test]
#[ignore = "build scripts/build-metal.sh --mode bitcoin first; requires Apple GPU"]
fn candidates_patterns_winners_and_errors_match_cpu() {
    super::support::address::check::<BitcoinTransport>(
        &artifacts(),
        super::support::address::Cases {
            seed: 10088153575472065218,
            patterns: [
                BytePattern::new(&[], &[]).unwrap(),
                BytePattern::new(&[b'q'; 43], &[]).unwrap(),
                BytePattern::new(b"bc1qg", b"6m").unwrap(),
                BytePattern::new(&[], b"q").unwrap(),
            ],
            address: logic::test_vectors::BECH32_P2WPKH_EXPECTED,
            winner_lane: 0,
            check_next_batch: false,
            extra_groups: &[],
        },
    );
}

#[test]
#[ignore = "build scripts/build-metal.sh --mode bitcoin first; requires Apple GPU"]
fn bounded_cli_prints_verified_winners_and_completes_misses() {
    for (prefix, suffix, matches) in [
        ("", "", true),
        (
            std::str::from_utf8(logic::test_vectors::BECH32_P2WPKH_EXPECTED).unwrap(),
            "",
            true,
        ),
        ("bc1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq", "", false),
    ] {
        let stdout = super::support::address::cli(
            &artifacts(),
            "bitcoin-vanity",
            10088153575472065218,
            prefix,
            suffix,
        );
        assert_eq!(stdout.contains("[bitcoin] wallet="), matches, "{stdout}");
        assert_eq!(stdout.contains("[bitcoin] address="), matches, "{stdout}");
        if prefix.starts_with("bc1qg") {
            assert!(
                stdout.contains("[bitcoin] address=bc1qgcz8ez3a3md3xnplrgl86edsl46zruf8mwx56m")
            );
        }
    }
}
